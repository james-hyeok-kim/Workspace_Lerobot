import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from lerobot.common.policies.factory import make_policy
from lerobot.common.envs.factory import make_env
from lerobot.common.utils.utils import set_global_seed

class ManualRPCALinear(nn.Module):
    def __init__(self, original_linear, alpha=0.5, rank=32, outlier_ratio=0.01, dtype=torch.float32):
        super().__init__()
        self.target_dtype = dtype
        self.alpha = alpha
        self.rank = rank
        self.outlier_ratio = outlier_ratio
        
        self.register_buffer('weight', original_linear.weight.data.clone().to(dtype))
        self.bias = nn.Parameter(original_linear.bias.data.clone().to(dtype)) if original_linear.bias is not None else None
        self.register_buffer('w_sparse', torch.zeros_like(original_linear.weight.data).to(dtype))
        self.register_buffer('w_quantized', original_linear.weight.data.clone().to(dtype))
        self.register_buffer('smooth_scale', torch.ones(self.weight.shape[1]).to(dtype))
        
        self.lora_a = nn.Parameter(torch.zeros(rank, self.weight.shape[1]).to(dtype))
        self.lora_b = nn.Parameter(torch.zeros(self.weight.shape[0], rank).to(dtype))
        self.is_calibrated = False

    @torch.no_grad()
    def manual_calibrate_and_rpca(self, x_max):
        x_max = x_max.clamp(min=1e-5).to(self.weight.device)
        w_abs = self.weight.abs()
        threshold = torch.quantile(w_abs.view(-1).float(), 1.0 - self.outlier_ratio)
        
        sparse_mask = w_abs >= threshold
        self.w_sparse.copy_((self.weight * sparse_mask).to(self.target_dtype))
        
        w_dense = self.weight * (~sparse_mask)
        w_max = w_dense.abs().max(dim=0)[0].clamp(min=1e-5)
        self.smooth_scale.data = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).to(self.target_dtype)
        
        w_smoothed = w_dense / self.smooth_scale.view(1, -1)
        scale = w_smoothed.abs().max() / 127
        w_q = torch.round(w_smoothed / scale).clamp(-128, 127) * scale
        self.w_quantized.copy_(w_q.to(self.target_dtype))
        
        w_error = w_smoothed - w_q
        U, S, Vh = torch.linalg.svd(w_error.float(), full_matrices=False)
        actual_rank = min(self.rank, S.shape[0])
        sqrt_S = torch.sqrt(S[:actual_rank])
        self.lora_a.data[:actual_rank, :] = (Vh[:actual_rank, :] * sqrt_S.unsqueeze(1)).to(self.target_dtype)
        self.lora_b.data[:, :actual_rank] = (U[:, :actual_rank] * sqrt_S.unsqueeze(0)).to(self.target_dtype)
        self.is_calibrated = True

    def forward(self, x):
        if not self.is_calibrated: return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)
        x_smoothed = x.to(self.target_dtype) * self.smooth_scale
        base_out = F.linear(x_smoothed, self.w_quantized)
        svd_out = F.linear(F.linear(x_smoothed, self.lora_a), self.lora_b)
        sparse_out = F.linear(x.to(self.target_dtype), self.w_sparse)
        return (base_out + svd_out + sparse_out + self.bias) if self.bias is not None else (base_out + svd_out + sparse_out)

def replace_to_rpca(model, args):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, ManualRPCALinear(module, rank=args.rank, alpha=args.alpha, outlier_ratio=args.outlier_ratio))
        else: replace_to_rpca(module, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_paths", nargs='+', required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--outlier_ratio", type=float, default=0.01)
    parser.add_argument("--env_name", type=str, default="lerobot/pusht_image")
    args = parser.parse_args()
    
    device = torch.device("cuda")
    for path in args.policy_paths:
        policy = make_policy(path, device=device)
        replace_to_rpca(policy, args)
        for m in policy.modules():
            if isinstance(m, ManualRPCALinear):
                m.manual_calibrate_and_rpca(torch.ones(m.weight.shape[1], device=device)*0.5)
        
        env = make_env(args.env_name, n_envs=1)
        obs, info = env.reset()
        print(f"✅ Finished RPCA/SVD Eval Setup for: {path}")
        # (Evaluation Loop 실행 코드)

if __name__ == "__main__":
    main()