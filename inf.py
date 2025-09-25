import os
import time
import pickle
import torch
import numpy as np
import tree
import argparse
import glob
import csv
import requests
import pandas as pd
import subprocess
import re
from torch.optim import Adam, AdamW
from diffusion import NoiseScheduleVP
from sampling import get_sampling_fn
from models.utils import create_model
from torch.utils.data import Dataset, DataLoader, random_split
from utils import get_data_inverse_scaler
from diffusion import NoiseScheduleVP
from models.GVP_diff import geo_batch, GVPTransCond
from models.ema import ExponentialMovingAverage
import torch.nn.functional as F
from config import get_config
from peptide_preprocessing_enhanced import peptide2data_enhanced as peptide2data
from peptide_preprocessing_enhanced import INDEX_TO_AA
import torch.nn.functional as F


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        return Adam(params, lr=config.optim.lr,
                    betas=(config.optim.beta1, 0.999),
                    eps=config.optim.eps,
                    weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        return AdamW(params, lr=config.optim.lr,
                     amsgrad=True,
                     weight_decay=1e-12)
    else:
        raise NotImplementedError(f"Optimizer {config.optim.optimizer} not supported!")


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def fold_sequence_with_esmfold(sequence, max_retries=3, delay=2):
    """
    Fold a protein sequence using ESM Atlas API
    """
    ESM_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    
    for attempt in range(max_retries):
        try:
            print(f"      Folding attempt {attempt + 1}/{max_retries}...", end="")
            
            response = requests.post(
                ESM_API_URL,
                data=sequence,
                headers={'Content-Type': 'text/plain'},
                timeout=60
            )
            
            if response.status_code == 200:
                print(" ✓")
                return response.text
            elif response.status_code == 429:
                print(f" Rate limited, waiting {delay * (attempt + 1)}s...")
                time.sleep(delay * (attempt + 1))
            else:
                print(f" API error: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    
        except requests.exceptions.Timeout:
            print(f" Timeout")
        except requests.exceptions.RequestException as e:
            print(f" Request failed: {e}")
            
        if attempt < max_retries - 1:
            time.sleep(delay)
    
    print(" ✗ Failed")
    return None


def calculate_tm_score(generated_pdb_content, original_pdb_path, temp_dir="/tmp"):
    """
    Calculate TM-score between generated structure and original PDB
    """
    try:
        # Create temporary file for generated structure
        temp_pdb = os.path.join(temp_dir, f"temp_generated_{int(time.time())}.pdb")
        with open(temp_pdb, 'w') as f:
            f.write(generated_pdb_content)
        
        # Set up environment
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/home/ribodiffusion/.conda/envs/ribodiffusionenv/lib:" + env.get("LD_LIBRARY_PATH", "")
        
        # Run TM-score
        command = subprocess.run(
            ["./TMscore", temp_pdb, original_pdb_path],
            capture_output=True,
            text=True,
            timeout=120,
            env=env
        )
        
        # Clean up temp file
        if os.path.exists(temp_pdb):
            os.remove(temp_pdb)
        
        if command.returncode == 0:
            output = command.stdout
            
            if "no common residues" in output.lower():
                return 0.0
            
            # Parse TM-score
            tm_score_match = re.search(r"TM-score\s*=\s*([\d.]+)", output, re.IGNORECASE)
            if tm_score_match:
                return float(tm_score_match.group(1))
            
            # Try alternative patterns
            patterns = [
                r"TM-score\s*:\s*([\d.]+)",
                r"TM\s*=\s*([\d.]+)",
                r"TM-score\s+([\d.]+)",
                r"([\d.]+)\s*\(normalized by length",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    return float(match.group(1))
        
        return None
        
    except Exception as e:
        print(f"      Error calculating TM-score: {str(e)}")
        return None


@torch.no_grad()
def vpsde_inference(config, pdb_file='original_pdbs/1lmw_A_peptide'):
    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]

    # === Model init / load ===
    model = GVPTransCond(config).to(config.device)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = get_optimizer(config, model.parameters())
    state = {'optimizer': optimizer, 'model': model, 'ema': ema, 'step': 0}

    ckpt = './ckpts/best_model.pth'
    loaded = torch.load(ckpt, map_location=config.device)
    ema.load_state_dict(loaded['ema'])
    ema.copy_to(model.parameters())
    model.to(config.device)

    # === Sampler ===
    noise_scheduler = NoiseScheduleVP(
        config.sde.schedule,
        continuous_beta_0=config.sde.continuous_beta_0,
        continuous_beta_1=config.sde.continuous_beta_1
    )
    inverse_scaler = lambda x: x
    sampling_fn = get_sampling_fn(
        config, noise_scheduler, config.eval.sampling_steps, inverse_scaler
    )

    # === Data prep ===
    print(f"\n🧬 Processing: {pdb_id}")
    struct_data = peptide2data(pdb_file,
                               num_posenc=config.data.num_posenc,
                               num_rbf=config.data.num_rbf,
                               knn_num=config.data.knn_num)
    struct_data['inference_mode'] = True
    original_seq = struct_data['seq'].clone().to(config.device)
    orig_str = ''.join([INDEX_TO_AA[int(a)] for a in original_seq])

    struct_data = tree.map_structure(
        lambda x: x.unsqueeze(0).repeat_interleave(config.eval.n_samples, dim=0).to(config.device)
        if isinstance(x, torch.Tensor) else x,
        struct_data
    )

    # === Inference ===
    print(f"📋 Generating {config.eval.n_samples} sequences...")
    start = time.time()
    samples_indices, samples_logits = sampling_fn(model, struct_data)
    print(f"✅ Done in {time.time() - start:.2f}s")

    # === Process sequences ===
    generated_sequences = []
    for i in range(config.eval.n_samples):
        seq_idx = samples_indices[i].detach().cpu()
        pred_str = ''.join([INDEX_TO_AA[int(a)] for a in seq_idx])
        generated_sequences.append(pred_str)

    print(f"\n📄 Original sequence: {orig_str} (length: {len(orig_str)})")
    print(f"📄 Generated {len(generated_sequences)} sequences")

    return {
        'pdb_id': pdb_id,
        'original_sequence': orig_str,
        'generated_sequences': generated_sequences,
        'original_pdb_path': pdb_file
    }


def complete_pipeline_display_only(pdb_file, config):
    """
    Complete pipeline with terminal display only (no file saving)
    """
    print("="*70)
    print("🧬 COMPLETE SINGLE PDB INFERENCE PIPELINE")
    print("="*70)
    
    # Step 1: Generate sequences
    print("\n📋 STEP 1: SEQUENCE GENERATION")
    print("-" * 50)
    result = vpsde_inference(config, pdb_file)
    
    pdb_id = result['pdb_id']
    original_seq = result['original_sequence']
    generated_sequences = result['generated_sequences']
    original_pdb_path = result['original_pdb_path']
    
    # Step 2: Fold sequences and calculate TM-scores
    print(f"\n🧪 STEP 2: FOLDING & TM-SCORE CALCULATION")
    print("-" * 50)
    
    results_data = []
    
    for i, sequence in enumerate(generated_sequences):
        print(f"\n  Sequence {i+1}/{len(generated_sequences)}: {sequence[:30]}... (len: {len(sequence)})")
        
        # Fold with ESMFold
        pdb_content = fold_sequence_with_esmfold(sequence)
        
        if pdb_content:
            print(f"      Calculating TM-score...", end="")
            tm_score = calculate_tm_score(pdb_content, original_pdb_path)
            
            if tm_score is not None:
                print(f" ✓ TM-score: {tm_score:.4f}")
            else:
                print(f" ✗ Failed")
                tm_score = 0.0
        else:
            tm_score = 0.0
        
        results_data.append({
            'sequence_index': i,
            'sequence': sequence,
            'sequence_length': len(sequence),
            'fold_successful': pdb_content is not None,
            'tm_score': tm_score
        })
        
        # Small delay between API calls
        time.sleep(1)
    
    # Step 3: Rank and display results
    print(f"\n🏆 STEP 3: FINAL RANKINGS")
    print("-" * 50)
    
    # Sort by TM-score (descending)
    results_data.sort(key=lambda x: x['tm_score'], reverse=True)
    
    # Display ranked results
    print(f"\n📊 RESULTS FOR {pdb_id}:")
    print("="*80)
    print(f"{'Rank':<4} {'Seq#':<4} {'TM-Score':<8} {'Length':<6} {'Folded':<7} {'Sequence':<50}")
    print("-"*80)
    
    for rank, result in enumerate(results_data, 1):
        seq_preview = result['sequence'][:47] + "..." if len(result['sequence']) > 50 else result['sequence']
        folded_status = "✓" if result['fold_successful'] else "✗"
        
        print(f"{rank:<4} {result['sequence_index']:<4} {result['tm_score']:<8.4f} "
              f"{result['sequence_length']:<6} {folded_status:<7} {seq_preview}")
    
    # Summary statistics
    successful_scores = [r['tm_score'] for r in results_data if r['fold_successful'] and r['tm_score'] > 0]
    successful_folds = len([r for r in results_data if r['fold_successful']])
    
    print(f"\n📊 SUMMARY:")
    print("-"*40)
    print(f"PDB ID: {pdb_id}")
    print(f"Original sequence length: {len(original_seq)}")
    print(f"Generated sequences: {len(generated_sequences)}")
    print(f"Successfully folded: {successful_folds}/{len(generated_sequences)}")
    print(f"TM-scores calculated: {len(successful_scores)}")
    
    if successful_scores:
        print(f"Best TM-score: {max(successful_scores):.4f}")
        print(f"Average TM-score: {np.mean(successful_scores):.4f}")
        print(f"Worst TM-score: {min(successful_scores):.4f}")
    else:
        print("No successful TM-score calculations")
    
    print(f"\n🎉 PIPELINE COMPLETE!")
    
    return results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single PDB inference with terminal display")
    parser.add_argument("--pdb_file", type=str, required=True,
                        help="Path to input PDB file")
    args = parser.parse_args()
    
    if not os.path.exists(args.pdb_file):
        print(f"Error: PDB file not found: {args.pdb_file}")
        exit(1)
    
    config = get_config()
    config.eval.n_samples = 10
    
    # Run complete pipeline (display only)
    complete_pipeline_display_only(args.pdb_file, config)