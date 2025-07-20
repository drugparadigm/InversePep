import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import os
import time
import random
import glob
import torch
import random
import tree
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from utils import get_data_inverse_scaler
from diffusion import NoiseScheduleVP
from models.GVP_diff import geo_batch, GVPTransCond
from models.ema import ExponentialMovingAverage
import torch.nn.functional as F
from config import get_config
from models.GVP_diff import geo_batch
from torch.optim.lr_scheduler import CosineAnnealingLR
from bucket_loader import LengthBucketedDataset, BucketedDataLoader

@torch.no_grad()
def compute_oracle_accuracy(model, struct_data, noise_scheduler, device, out_dim):
    """
    Compute batch oracle accuracy and count of perfectly matched sequences.

    Returns:
        perfect_match_ratio: float (0.0 to 1.0)
        avg_oracle_accuracy: float (0.0 to 1.0)
    """
    B = struct_data['seq'].shape[0]

    # Send batch to device
    struct_data = {k: v.to(device) for k, v in struct_data.items()}

    # One-hot encode the ground truth
    S0 = F.one_hot(struct_data['seq'], num_classes=out_dim).float()

    # Sample noise
    eps = 1e-5
    t = torch.rand(B, device=device) * (noise_scheduler.T - 2 * eps) + eps
    alpha_t, sigma_t = noise_scheduler.marginal_prob(t)
    noise = torch.randn_like(S0)
    St = alpha_t.view(-1, 1, 1) * S0 + sigma_t.view(-1, 1, 1) * noise
    struct_data['z_t'] = St

    lambda_t = (alpha_t ** 2 / sigma_t ** 2).log()
    pred = model(struct_data, cond_drop_prob=0.0, cond_x=None, noise_level=lambda_t)  # [B, L, D]

    pred_ids = pred.argmax(dim=-1)  # [B, L]
    true_ids = struct_data['seq']   # [B, L]

    pad_token = 21
    mask = (true_ids != pad_token).float()  # [B, L]

    
    matches = (pred_ids == true_ids).float() * mask
    seq_equal = ((matches.sum(dim=1) == mask.sum(dim=1)) & (mask.sum(dim=1) > 0)).float()
    perfectly_matched = seq_equal.sum().item()
    perfect_match_ratio = perfectly_matched / B

  
    token_acc = (matches.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8))  # [B]
    avg_oracle_accuracy = token_acc.mean().item()

    return perfect_match_ratio, avg_oracle_accuracy

def set_random_seed(config):
    seed = config.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(config, params):
    """Return a flax optimizer object based on config."""
    if config.optim.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=config.optim.lr, amsgrad=True, weight_decay=1e-12)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!'
        )
    return optimizer

def collate_fn(batch_dicts):
    """
    Batch a list of raw graph dicts into batched dict of tensors,
    padding node-level and edge-level arrays to uniform sizes.
    """
    B = len(batch_dicts)
    Nmax = max(d['node_s'].shape[0] for d in batch_dicts)
    Emax = max(d['edge_s'].shape[0] for d in batch_dicts)
    
    batched = {}
    
    # Node-level keys
    for key in ['seq', 'coords', 'node_s', 'node_v', 'z_t', 'mask']:
        samples = []
        for d in batch_dicts:
            t = d[key]
            Ni = t.shape[0]
            if Ni < Nmax:
                pad_shape = (Nmax - Ni,) + t.shape[1:]
                if key == 'seq':
                    pad = torch.full(pad_shape, 21, dtype=t.dtype, device=t.device)
                else:
                    pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
                t = torch.cat([t, pad], dim=0)
            samples.append(t.unsqueeze(0))
        batched[key] = torch.cat(samples, dim=0)
    
    # Edge index
    samples_idx = []
    for d in batch_dicts:
        idx = d['edge_index']
        Ei = idx.shape[1]
        if Ei < Emax:
            pad_cnt = Emax - Ei
            pad = torch.zeros((2, pad_cnt), dtype=idx.dtype, device=idx.device)
            idx = torch.cat([idx, pad], dim=1)
        samples_idx.append(idx.unsqueeze(0))
    batched['edge_index'] = torch.cat(samples_idx, dim=0)
    
    # Edge scalar features
    samples_s = []
    for d in batch_dicts:
        s = d['edge_s']
        Ei = s.shape[0]
        if Ei < Emax:
            pad_shape = (Emax - Ei,) + s.shape[1:]
            pad = torch.zeros(pad_shape, dtype=s.dtype, device=s.device)
            s = torch.cat([s, pad], dim=0)
        samples_s.append(s.unsqueeze(0))
    batched['edge_s'] = torch.cat(samples_s, dim=0)
    
    # Edge vector features
    samples_v = []
    for d in batch_dicts:
        v = d['edge_v']
        Ei = v.shape[0]
        if Ei < Emax:
            pad_shape = (Emax - Ei,) + v.shape[1:]
            pad = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
            v = torch.cat([v, pad], dim=0)
        samples_v.append(v.unsqueeze(0))
    batched['edge_v'] = torch.cat(samples_v, dim=0)
    
    return batched

def print_avg_length(files, label):
    lengths = [torch.load(f)['seq'].shape[0] for f in files]
    print(f"{label} set: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

class PtDataset(Dataset):
    """
    Loads pre-processed graph dicts saved as .pt files from a directory.
    """
    def __init__(self, pt_folder):
        self.files = sorted(glob.glob(os.path.join(pt_folder, '*.pt')))
        if not self.files:
            raise ValueError(f"No .pt files found in {pt_folder}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        g = torch.load(self.files[idx])
        seq_len = g['seq'].shape[0]
        g['z_t'] = torch.zeros((seq_len, 22), dtype=torch.float32)
        return g

# ---- Training function ----
def train(config, pt_folder):
    # 1) seeds & device
    set_random_seed(config)
    device = config.device
  
    config.model.self_cond         = True
    config.train.drop_struct_prob  = 0.5

    # 2) model, optimizer, EMA
    model     = GVPTransCond(config).to(device)
    model.train()
    optimizer = get_optimizer(config, model.parameters())
   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.optim.lr_decay_epochs,
    eta_min=config.optim.lr_min
    )
    ema       = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)

    # 3) noise scheduler & scaler
    noise_scheduler = NoiseScheduleVP(
        config.sde.schedule,
        continuous_beta_0=config.sde.continuous_beta_0,
        continuous_beta_1=config.sde.continuous_beta_1
    )
    inverse_scaler = get_data_inverse_scaler(config)

    # 4) dataset & train/val/test split
    full_ds = LengthBucketedDataset(pt_folder, bucket_size=5)
      
    file_list = sorted(glob.glob(os.path.join(pt_folder, '*.pt')))

    random.seed(config.seed)
    random.shuffle(file_list)
    N = len(file_list)
    n_train = int(0.8 * N)
    n_val   = int(0.1 * N)
    n_test  = N - n_train - n_val

    train_files = file_list[:n_train]
    val_files   = file_list[n_train:n_train+n_val]
    test_files  = file_list[n_train+n_val:]

    print_avg_length(train_files, "Train")
    print_avg_length(val_files, "Val")
    print_avg_length(test_files, "Test")
    train_ds = LengthBucketedDataset(train_files, bucket_size=5)
    val_ds   = LengthBucketedDataset(val_files, bucket_size=5)
    test_ds  = LengthBucketedDataset(test_files, bucket_size=5)

    train_loader = BucketedDataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = BucketedDataLoader(val_ds, batch_size=config.train.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = BucketedDataLoader(test_ds, batch_size=config.train.batch_size, shuffle=False, collate_fn=collate_fn)
    global_step = 0
    best_val_loss = float('inf')
    best_ckpt_path = os.path.join(config.train.ckpt_dir, 'best_model.pth')
    for epoch in range(config.train.epochs):
        # training
        epoch_loss, t0 = 0.0, time.time()
        print(f"\n🔁 Starting Epoch {epoch} with LR: {scheduler.get_last_lr()[0]:.6e}")
        for struct_data in train_loader:
            true_seq = struct_data['seq']
            struct_data = tree.map_structure(lambda x: x.to(device), struct_data)
            S0 = F.one_hot(struct_data['seq'], num_classes=config.model.out_dim).float()
            B, Nn, D = S0.shape
            eps = 1e-5
            T = noise_scheduler.T
            t = torch.rand(B, device=device) * (T - 2*eps) + eps
            alpha_t, sigma_t = noise_scheduler.marginal_prob(t)
            noise = torch.randn_like(S0) 
            St = alpha_t.view(-1,1,1)*S0 + sigma_t.view(-1,1,1)*noise
            lambda_t = (alpha_t**2 / sigma_t**2).log()
            if random.random() < config.train.self_cond_prob: 
                with torch.no_grad():
                    tmp = dict(struct_data)
                    tmp['z_t'] = St
                    prev = model(tmp, cond_drop_prob=config.train.drop_struct_prob, cond_x=None, noise_level=lambda_t)
                cond_x = prev.detach()
            else:
                cond_x = None
            struct_data['z_t'] = St
            pred = model(struct_data, cond_drop_prob=0.0, cond_x=cond_x, noise_level=lambda_t) 
            w = torch.sqrt(alpha_t / sigma_t).view(-1,1,1).clamp(max=100)
            mask = struct_data['mask'].unsqueeze(-1).float()
            mse = ((w * (pred - S0))**2 * mask).sum() / mask.sum() / D
            ce_loss_per_token = F.cross_entropy(pred.transpose(1,2), struct_data['seq'], reduction='none')
            nonpad_mask = (struct_data['seq'] != 21).float()
            ce = (ce_loss_per_token * nonpad_mask).sum() / nonpad_mask.sum()
            loss = 0.3 * mse + 0.7 * ce

            
            optimizer.zero_grad() 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model.parameters())
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % config.train.log_every == 0:
                with torch.no_grad():
                  acc = (pred.argmax(-1) == struct_data['seq']).float().mean().item()
                  
                  perfect_match_ratio, avg_oracle_accuracy = compute_oracle_accuracy(model, struct_data, noise_scheduler, device,config.model.out_dim)
                lengths = (struct_data['mask'].sum(dim=1)).tolist()
                print("------------------------------------------------------------------------------------------------------------------------------------------------\n")
                print(f"[Step {global_step}] Batch sequence lengths: {lengths}")
                print(f"[Step {global_step}] train_loss = {loss.item():.4e} perfect_match_ratio={perfect_match_ratio:.2%} avg_oracle_accuracy={avg_oracle_accuracy:.2%}")
                
                IDX_TO_AA = [
                'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', '-'
                ]
                pred_seq = pred.argmax(-1)[0]     # [L]
                true_seq = struct_data['seq'][0]  # [L]

               
                mask = (true_seq != 21)
                pred_aa = ''.join(IDX_TO_AA[int(i)] for i in pred_seq[mask])
                true_aa = ''.join(IDX_TO_AA[int(i)] for i in true_seq[mask])
                
                print(f"True seq : {true_aa}")
                print(f"Pred seq : {pred_aa}")

                    
        train_avg = epoch_loss / len(train_loader)
        
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for struct_data in val_loader:
                struct_data = tree.map_structure(lambda x: x.to(device), struct_data)
                S0 = F.one_hot(struct_data['seq'], num_classes=config.model.out_dim).float()
                B, Nn, D = S0.shape
                eps = 1e-5
                T = noise_scheduler.T
                t = torch.rand(B, device=device) * (T - 2*eps) + eps
                alpha_t, sigma_t = noise_scheduler.marginal_prob(t)
                noise = torch.randn_like(S0)
                St = alpha_t.view(-1,1,1)*S0 + sigma_t.view(-1,1,1)*noise
                lambda_t = (alpha_t**2 / sigma_t**2).log()
                struct_data['z_t'] = St
                pred = model(struct_data, cond_drop_prob=0.0, cond_x=None, noise_level=lambda_t)
                w = torch.sqrt(alpha_t / sigma_t).view(-1,1,1).clamp(max=100)
                mask = struct_data['mask'].unsqueeze(-1).float()
                mse = ((w * (pred - S0))**2 * mask).sum() / mask.sum() / D
                ce_loss_per_token = F.cross_entropy(pred.transpose(1,2), struct_data['seq'], reduction='none')
                nonpad_mask = (struct_data['seq'] != 21).float()
                ce = (ce_loss_per_token * nonpad_mask).sum() / nonpad_mask.sum()
                val_loss = 0.3 * mse + 0.7 * ce
        val_avg = val_loss / len(val_loader)

        print(f"Epoch {epoch:03d} | train_loss = {train_avg:.4e} | val_loss = {val_avg:.4e} | time {time.time()-t0:.1f}s")
        os.makedirs(config.train.ckpt_dir, exist_ok=True)
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save({
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'ema':   ema.state_dict(),
                'step':  global_step
            }, best_ckpt_path)
            print(f"New best model saved (val_loss={best_val_loss:.4e})")
        model.train()
        if (epoch+1) % config.train.save_every == 0:
            os.makedirs(config.train.ckpt_dir, exist_ok=True)
            ckpt = {
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'ema':   ema.state_dict(),
                'step':  global_step
            }
            torch.save(ckpt, os.path.join(config.train.ckpt_dir, f'epoch_{epoch+1}.pth'))
        
        scheduler.step()
        print(f"📉 Updated learning rate after epoch {epoch}: {scheduler.get_last_lr()[0]:.6e}")

    # ---- testing ----
    model.eval()
    test_loss = 0.

    if os.path.exists(best_ckpt_path):
        print(f"Loading best model from {best_ckpt_path} for final test")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
    else:
        print("Warning: best_model.pth not found, using final epoch model")
    with torch.no_grad():
        for struct_data in test_loader:
            struct_data = tree.map_structure(lambda x: x.to(device), struct_data)
            S0 = F.one_hot(struct_data['seq'], num_classes=config.model.out_dim).float()
            B, Nn, D = S0.shape
            eps = 1e-5
            T = noise_scheduler.T
            t = torch.rand(B, device=device) * (T - 2*eps) + eps
            alpha_t, sigma_t = noise_scheduler.marginal_prob(t)
            noise = torch.randn_like(S0)
            St = alpha_t.view(-1,1,1)*S0 + sigma_t.view(-1,1,1)*noise
            lambda_t = (alpha_t**2 / sigma_t**2).log()
            struct_data['z_t'] = St
            pred = model(struct_data, cond_drop_prob=0.0, cond_x=None, noise_level=lambda_t)
            w = torch.sqrt(alpha_t / sigma_t).view(-1,1,1).clamp(max=100)
            mask = struct_data['mask'].unsqueeze(-1).float()
            mse = ((w * (pred - S0))**2 * mask).sum() / mask.sum() / D
            ce_loss_per_token = F.cross_entropy(pred.transpose(1,2), struct_data['seq'], reduction='none')
            nonpad_mask = (struct_data['seq'] != 21).float()
            ce = (ce_loss_per_token * nonpad_mask).sum() / nonpad_mask.sum()
            test_loss = 0.3 * mse + 0.7 * ce
    test_avg = test_loss / len(test_loader)
    print(f"Final test_loss = {test_avg:.4e}")

    print("✅ Training complete.")

if __name__ == '__main__':
    cfg = get_config()
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
       print(f"  GPU {i} ↦", torch.cuda.get_device_name(i))
    train(cfg, 'peptides_pt_enhanced_latest_3')
