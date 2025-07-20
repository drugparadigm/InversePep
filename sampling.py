import numpy as np
import torch
from torch.nn import functional as F
import random
import pdb
from torch_geometric.data import Batch
import utils


def post_process(gen_seq, inverse_scaler,temperature=0.8,top_k=None):
    """Post process generated sequences."""
    gen_seq = inverse_scaler(gen_seq)
    gen_seq = torch.argmax(gen_seq, dim=-1)
    return gen_seq

    
def get_sampling_fn(config, noise_scheduler, sampling_steps, inverse_scaler, eps=1e-3):
    device = config.device
    D = config.model.out_dim  

    time_steps = torch.linspace(noise_scheduler.T, eps, sampling_steps, device=device)
    sampler = AncestralSampler(config, noise_scheduler, time_steps, config.model.pred_data, inverse_scaler)

    def sampling_fn(model, data):
        model.eval()
        batch = data
        
        if 'z_t' not in batch or batch.get('inference_mode', True):
            if 'z_t' in batch:
                target_shape = batch['z_t'].shape
            else:
                B = batch['seq'].shape[0]
                L = batch['seq'].shape[1]
                target_shape = (B, L, D)
            
            # Start from random noise for generation
            z = torch.randn(target_shape, device=device)
        else:
            z = torch.randn_like(batch['z_t'])
            
        gen_seq = sampler.sampling(model, z, batch)
        gen_seqs = post_process(gen_seq, inverse_scaler, temperature=0.8, top_k=8)
        return gen_seqs
    return sampling_fn

def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


class AncestralSampler:
    """Ancestral sampling for RNA inverse design."""
    def __init__(self, config, noise_scheduler, time_steps, model_pred_data, inverse_scaler):
        self.noise_scheduler = noise_scheduler
        self.t_array = time_steps
        self.s_array = torch.cat([time_steps[1:], torch.zeros(1, device=time_steps.device)])
        self.model_pred_data = model_pred_data
        self.self_cond = config.model.self_cond
        self.cond_scale = getattr(config.eval, 'cond_scale', -1.)
        self.dynamic_threshold = getattr(config.eval, 'dynamic_threshold', False)
        self.dynamic_thresholding_percentile = getattr(config.eval, 'dynamic_thresholding_percentile', 0.95)

    @torch.no_grad()
    def forward_with_cond_scale(self, model, *args, cond_scale=1.0, **kwargs):
        logits = model(*args, **kwargs)  # with condition
        if cond_scale == 1.0:
            return logits

        null_logits = model(*args, cond_drop_prob=1.0, **kwargs)  # without condition
        return null_logits + (logits - null_logits) * cond_scale

    @torch.no_grad()
    def sampling(self, model, z_T, batch):
        batch['z_t'] = x = z_T
        bs = z_T.shape[0]
        cond_x = None
        for i in range(len(self.t_array)):
            t = self.t_array[i]
            s = self.s_array[i]

            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t

            # vec_t = torch.ones(bs, device=x.device) * t
            # noise_level = torch.ones(bs, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)
            noise_level = torch.log(alpha_t ** 2 / sigma_t ** 2)

            pred_t = model(batch, time=t.unsqueeze(0), noise_level=noise_level.unsqueeze(0), cond_x=cond_x) \
                if self.cond_scale < 0. \
                else self.forward_with_cond_scale(model, batch, cond_scale=self.cond_scale, time=t.unsqueeze(0),
                                                 noise_level=noise_level.unsqueeze(0), cond_x=cond_x)

            # dynamic thresholding
            if self.dynamic_threshold:
                # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
                s = torch.quantile(
                    pred_t.reshape(bs, -1).abs(),
                    self.dynamic_thresholding_percentile,
                    dim=-1
                )
                s.clamp_(min=1.)

                s = expand_dims(s, pred_t.dim())
                pred_t = pred_t.clamp(-s, s) / s

            if self.self_cond:
                assert self.model_pred_data
                cond_x = pred_t.detach().clone()

            # seq update
            if pred_t.shape != x.shape:
                pred_t = pred_t.unsqueeze(-2)

            if self.model_pred_data:
                x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                         + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), pred_t.dim()) * pred_t
            else:
                x_mean = x / expand_dims(alpha_t_given_s.repeat(bs), x.dim()) \
                         - expand_dims((sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), pred_t.dim()) * pred_t

            batch['z_t'] = x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * \
                torch.randn(x_mean.shape, device=x.device)
        return x_mean.squeeze(-2)