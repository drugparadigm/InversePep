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
import torch
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

# ---------- Lazy loading for ESMFold -------------
_esmfold_singleton = {"esm_model": None, "tokenizer": None, "device": None}

def get_esmfold():
    if _esmfold_singleton["esm_model"] is None:
        from transformers import AutoTokenizer, EsmForProteinFolding
        from accelerate.test_utils.testing import get_backend

        DEVICE, _, _ = get_backend()
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        esm_model = esm_model.to(DEVICE)

        _esmfold_singleton["esm_model"] = esm_model
        _esmfold_singleton["tokenizer"] = tokenizer
        _esmfold_singleton["device"] = DEVICE

        print(f"Model loaded on {DEVICE}")
    
    return _esmfold_singleton["esm_model"], _esmfold_singleton["tokenizer"], _esmfold_singleton["device"]

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

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def fold_sequence_with_esmfold(sequence: str, save_path: str = None):
    """
    Fold a single protein sequence locally using ESMFold (Hugging Face).
    
    Args:
        sequence (str): Amino acid sequence.
        save_path (str, optional): Path to save the PDB. If None, it won't save.
    
    Returns:
        List[str]: List of PDB strings (usually length 1 for a single sequence).
    """
    esm_model, tokenizer, DEVICE = get_esmfold()  # lazy load on first call

    tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
    tokenized_input = tokenized_input.to(DEVICE)
    
    esm_model.eval()
    with torch.no_grad():
        outputs = esm_model(tokenized_input)
    
    pdbs = convert_outputs_to_pdb(outputs)
    
    if save_path:
        with open(save_path, "w") as f:
            f.write("".join(pdbs))
    
    return pdbs


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

        print(command.stderr)
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
def vpsde_inference(config, pdb_file):
    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]

    # === Model init / load ===
    model = GVPTransCond(config).to(config.device)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = get_optimizer(config, model.parameters())
    state = {'optimizer': optimizer, 'model': model, 'ema': ema, 'step': 0}

    # ckpt = './ckpts/best_model.pth'
    # ckpt = './ckpts/ablasation_study_1.pth'
    ckpt = './ckpts/ablation_study_2.pth'
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
    samples_indices = sampling_fn(model, struct_data)
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
    # Step 1: Generate sequences
    print("\n📋 STEP 1: SEQUENCE GENERATION")
    result = vpsde_inference(config, pdb_file)
    
    pdb_id = result['pdb_id']
    original_seq = result['original_sequence']
    generated_sequences = result['generated_sequences']
    original_pdb_path = result['original_pdb_path']
    
    # Step 2: Fold sequences and calculate TM-scores
    print(f"\n🧪 STEP 2: FOLDING & TM-SCORE CALCULATION")    
    results_data = []
    
    for i, sequence in enumerate(generated_sequences):
        print(f"\n  Sequence {i+1}/{len(generated_sequences)}: {sequence[:30]}... (len: {len(sequence)})")
        
        # Fold with ESMFold
        pdb_content_list = fold_sequence_with_esmfold(sequence)
        pdb_content = "".join(pdb_content_list)  # <- convert list to single string
        if pdb_content:
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
