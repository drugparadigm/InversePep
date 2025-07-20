import os
import time
import pickle
import torch
import numpy as np
import tree
import argparse
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
from peptide_preproocessing_enhanced import peptide2data_enhanced as peptide2data
from peptide_preproocessing_enhanced import INDEX_TO_AA

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


def evaluate_peptide_quality(generated_seq, original_seq):
    recovery = generated_seq.eq(original_seq).float().mean().item()
    return {'sequence_recovery': recovery}


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


@torch.no_grad()
def vpsde_inference(config, save_folder, pdb_file='example/1AMB.pdb'):
    os.makedirs(save_folder, exist_ok=True)

    # Model init
    model = GVPTransCond(config).to(config.device)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = get_optimizer(config, model.parameters())
    state = {'optimizer': optimizer, 'model': model, 'ema': ema, 'step': 0}

    # Load checkpoint
    ckpt = './ckpts/best_model.pth'
    loaded = torch.load(ckpt, map_location=config.device)
    # model.load_state_dict(loaded['model'], strict=False)
    missing, unexpected = model.load_state_dict(loaded['model'])
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    ema.load_state_dict(loaded['ema'])
    ema.copy_to(model.parameters())
    model.to(config.device)
    

    # Noise scheduler & sampling
    noise_scheduler = NoiseScheduleVP(
        config.sde.schedule,
        continuous_beta_0=config.sde.continuous_beta_0,
        continuous_beta_1=config.sde.continuous_beta_1
    )
    inverse_scaler = lambda x: x
    sampling_fn = get_sampling_fn(
        config, noise_scheduler, config.eval.sampling_steps, inverse_scaler
    )

    # Prepare data
    struct_data = peptide2data(pdb_file,
                               num_posenc=config.data.num_posenc,
                               num_rbf=config.data.num_rbf,
                               knn_num=config.data.knn_num)
    struct_data['inference_mode'] = True
    
 
    original_seq = struct_data['seq'].clone()
    original_seq = original_seq.to(config.device)
    orig_str = ''.join([INDEX_TO_AA[int(a)] for a in original_seq])

 
    struct_data = tree.map_structure(
    lambda x: x.unsqueeze(0).repeat_interleave(config.eval.n_samples, dim=0).to(config.device)
    if isinstance(x, torch.Tensor) else x,
    struct_data
    )

    # Inference
    print(f"Original sequence: {orig_str}")
    print(f"Generating {config.eval.n_samples} peptide designs for {pdb_file}")
    start = time.time()
    samples = sampling_fn(model, struct_data)
    duration = time.time() - start
    print(f"Done in {duration:.2f}s")

    # Setup output dirs
    fasta_dir = os.path.join(save_folder, 'fasta')
    metrics_dir = os.path.join(save_folder, 'metrics')
    compare_file = os.path.join(save_folder, 'comparison.txt')
    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    all_metrics = []
    # Write header for comparison
    with open(compare_file, 'w') as cf:
        cf.write(f"Original: {orig_str}\n\n")
    
    for i, seq in enumerate(samples):
        # convert to string
        pred_str = ''.join([INDEX_TO_AA[int(a)] for a in seq])
        # save FASTA
        fasta_path = os.path.join(fasta_dir, f"sample_{i}.fasta")
        with open(fasta_path, 'w') as f:
            f.write(f">sample_{i}\n" + pred_str)

        # record comparison
        with open(compare_file, 'a') as cf:
            cf.write(f"Sample {i}: {pred_str}\n")

        # metrics
        metrics = evaluate_peptide_quality(seq, original_seq)
        all_metrics.append(metrics)
        print(f"Sample {i}: recovery={metrics['sequence_recovery']:.4f}")
        print(f"Predicted: {pred_str}\n")

    # Avg metrics
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    with open(os.path.join(metrics_dir, 'avg_metrics.pkl'), 'wb') as f:
        pickle.dump({'individual': all_metrics, 'average': avg_metrics}, f)

    print("✅ Inference complete")
    print(f"Comparison saved to {compare_file}")
    # Combine labels and sequences into a list of strings
    all_seqs = [orig_str] + [''.join([INDEX_TO_AA[int(a)] for a in seq]) for seq in samples]
    sequence_labels = ["Original"] + [f"Sample {i}" for i in range(len(samples))]

    # Make a list of formatted strings
    sequence_list = [orig_str] + [''.join([INDEX_TO_AA[int(a)] for a in seq]) for seq in samples]
    # Print the list
    print("\n🧬 Final Sequence List:")
    print(sequence_list)

    return avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, default='results',
                         help="Directory to save outputs")
    parser.add_argument("--pdb_file", type=str, default='example/1AMB.pdb',
                         help="Path to the input PDB file")
    args = parser.parse_args()

    config = get_config()
 

    vpsde_inference(config, args.save_folder)
