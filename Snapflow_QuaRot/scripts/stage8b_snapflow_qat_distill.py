"""Stage 8b — SnapFlow QAT distillation, INT4 g=16 model (improved LR).

stage8 (LR=1e-5) 결과: 학습이 거의 이루어지지 않음 (max_diff<0.005).
원인: LR=1e-5는 bfloat16 weight 정밀도 대비 너무 작아 effective update ≈ 0.

개선:
  - LR = 5e-4 (50배 상향)
  - warmup 500 steps → cosine annealing
  - stage1_student warmstart (pre-rotation), resume 없이 fresh start
  - batch_size=4 per GPU, single GPU

Run:
    CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 python \\
        scripts/stage8b_snapflow_qat_distill.py \\
        --train_steps 5000 --batch_size 4 --lr 5e-4 --save_every 500
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in [
    str(_ROOT / "Snapflow_QuaRot"),
    str(_ROOT / "lerobot" / "src"),
    str(_ROOT / "TensorRT-Model-Optimizer"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

PRETRAINED_PATH = "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
STUDENT_CKPT    = "artifacts/stage1_student.safetensors"
NORMALIZER_STATS_PATH = (
    "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
    "/policy_preprocessor_step_2_normalizer_processor.safetensors"
)
DATASET_PATH    = "/data/jameskimh/james_libero_datasets/libero_10"
OUTPUT_PATH     = "artifacts/stage8b_student.safetensors"
OHB_MANIFEST_PATH = "artifacts/stage4_ohb_manifest.json"
_R3_SEED        = 100


def _load_policy(pretrained_path: str, device: torch.device):
    import torch._dynamo as _dynamo
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy
    from lerobot.envs.configs import LiberoEnv
    _dynamo.reset()
    cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    cfg.pretrained_path = pretrained_path
    cfg.device = str(device)
    cfg.use_amp = False
    cfg.compile_model = False
    cfg.gradient_checkpointing = False
    cfg.n_action_steps = 10
    cfg.num_inference_steps = 10
    env_cfg = LiberoEnv(task="libero_10", task_ids=list(range(10)))
    return make_policy(cfg=cfg, env_cfg=env_cfg).to(device)


def _patch_embed_image(policy):
    pw = policy.model.paligemma_with_expert
    def pe(image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        p = pw.paligemma
        out = p.model.get_image_features(image)
        f = (out.pooler_output if hasattr(out, "pooler_output") else out) * p.config.text_config.hidden_size ** 0.5
        return f.to(out_dtype)
    pw.embed_image = pe


def _apply_quarot(policy, device: torch.device):
    from quarot.fuse_rmsnorm import fuse_all_rmsnorms
    from quarot.offline_rotate import apply_r1r2r3, _make_signs, _apply_r3_to_layer
    from quarot.rotations import hadamard_transform

    inner = policy.model
    pali  = inner.paligemma_with_expert
    llm   = pali.paligemma.model.language_model
    expert = pali.gemma_expert
    llm_layers    = list(llm.layers)
    expert_layers = list(expert.model.layers)
    gdev = str(device)

    fuse_all_rmsnorms(llm, scope="llm")
    fuse_all_rmsnorms(expert, scope="dit")

    D = _make_signs(llm_layers[0].self_attn.q_proj.weight.shape[1],
                    device=gdev,
                    generator=torch.Generator(device=gdev).manual_seed(42)).float()

    emb = llm.embed_tokens.weight
    emb.data = (hadamard_transform(emb.data.float(), rotate_fp32=True) * D[None, :].to(emb.device)).to(emb.dtype)

    mmp = pali.paligemma.model.multi_modal_projector.linear
    W = hadamard_transform(mmp.weight.data.float().T, rotate_fp32=True).T * D[:, None].to(mmp.weight.device)
    mmp.weight.data = W.to(mmp.weight.dtype)
    if mmp.bias is not None:
        b = hadamard_transform(mmp.bias.data.float().unsqueeze(0), rotate_fp32=True).squeeze(0) * D.to(mmp.bias.device)
        mmp.bias.data = b.to(mmp.bias.dtype)

    apply_r1r2r3(llm_layers, r1=True, r2=False, r3=False, device=gdev, seed=42)

    gen_r3 = torch.Generator(device=gdev).manual_seed(_R3_SEED)
    signs  = _make_signs(llm_layers[0].self_attn.head_dim, device=gdev, generator=gen_r3).float()
    with torch.no_grad():
        for layer in llm_layers:   _apply_r3_to_layer(layer, signs)
        for layer in expert_layers: _apply_r3_to_layer(layer, signs)
    log.info("R1+R3 applied")


def _apply_int4_g16(policy, device: torch.device):
    import modelopt.torch.quantization as mtq
    from quant.w4a4_recipe import build_int4_weight_only_config

    ohb_manifest = {}
    if Path(OHB_MANIFEST_PATH).exists():
        with open(OHB_MANIFEST_PATH) as f:
            ohb_manifest = json.load(f)

    quant_cfg = build_int4_weight_only_config(
        group_size=16, ohb_manifest=ohb_manifest, algorithm="max"
    )

    def forward_loop(model):
        from snapflow.data import LiberoHDF5Dataset
        from snapflow.distill_loss import get_velocity
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        from torch.utils.data import DataLoader
        model.eval()
        dataset = LiberoHDF5Dataset(
            dataset_path=DATASET_PATH,
            normalizer_stats_path=NORMALIZER_STATS_PATH,
            seed=0,
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        n = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                try:
                    images, img_masks = model._preprocess_images(batch)
                    tokens = batch[OBS_LANGUAGE_TOKENS]
                    masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]
                    actions = model.prepare_action(batch)
                    noise = model.model.sample_noise(actions.shape, device)
                    t = torch.ones(actions.shape[0], device=device)
                    get_velocity(model.model, images, img_masks, tokens, masks, noise, t, s=t)
                except Exception:
                    pass
                n += 1
                if n >= 64:
                    break
        log.info(f"Calibration: {n} batches")

    mtq.quantize(policy, quant_cfg, forward_loop=forward_loop)
    log.info(f"INT4 g=16 applied (OHB: {len(ohb_manifest)} protected)")


def _freeze_vlm_unfreeze_expert(policy):
    for p in policy.parameters():
        p.requires_grad_(False)

    inner = policy.model
    for m in [
        inner.paligemma_with_expert.gemma_expert,
        inner.action_in_proj,
        inner.action_out_proj,
    ]:
        for p in m.parameters():
            p.requires_grad_(True)

    n_total = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    log.info(f"Frozen. Trainable: {n_train:,}/{n_total:,} ({100.*n_train/n_total:.1f}%)")


def _save(policy, path: Path):
    from safetensors.torch import save_file
    state = {k: v.contiguous().cpu() for k, v in policy.state_dict().items()}
    save_file(state, str(path))


def train(args):
    device = torch.device(f"cuda:{args.local_gpu}")
    torch.cuda.set_device(device)

    log.info("Loading policy...")
    policy = _load_policy(args.pretrained_path, device)
    _patch_embed_image(policy)

    # Warmstart from stage1_student (pre-rotation)
    if Path(args.warmstart_ckpt).exists():
        from safetensors.torch import load_file
        state = load_file(args.warmstart_ckpt)
        state = {k: v for k, v in state.items() if not k.startswith("target_time_emb.")}
        missing, unexpected = policy.load_state_dict(state, strict=False)
        log.info(f"Warmstart loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        log.warning(f"Warmstart not found: {args.warmstart_ckpt}")

    # Rotation → Quantization (calibrated on stage1_student rotated)
    log.info("Applying R1+R3 rotation...")
    _apply_quarot(policy, device)

    log.info("Applying INT4 g=16 (calibrated on stage1_student rotated)...")
    _apply_int4_g16(policy, device)

    # Freeze VLM, unfreeze expert
    _freeze_vlm_unfreeze_expert(policy)

    # Data
    from snapflow.data import LiberoHDF5Dataset
    from snapflow.distill_loss import SnapFlowLoss
    from torch.utils.data import DataLoader

    dataset = LiberoHDF5Dataset(
        dataset_path=args.dataset_path,
        normalizer_stats_path=args.normalizer_stats_path,
        seed=args.seed,
    )
    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True,
                        generator=g, persistent_workers=(args.num_workers > 0))

    # Optimizer with warmup + cosine
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    warmup_steps = args.warmup_steps
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=args.train_steps - warmup_steps, eta_min=args.lr * 0.05)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    loss_fn = SnapFlowLoss(action_dim=7, alpha=1.0, lam=0.0)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Training {args.train_steps} steps, batch={args.batch_size}, lr={args.lr}, warmup={warmup_steps}")
    log.info(f"Dataset: {len(dataset)} samples, trainable params: {sum(p.numel() for p in trainable_params):,}")

    loader_iter = iter(loader)
    step_times = []
    w0 = policy.model.action_in_proj.weight.data.clone()  # track actual weight change

    for step in range(args.train_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
                 for k, v in batch.items()}

        t0 = time.perf_counter()
        policy.train()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss_dict = loss_fn(policy, batch)

        loss = loss_dict["loss"]
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        step_times.append(time.perf_counter() - t0)

        if step % 100 == 0:
            avg_t = sum(step_times[-50:]) / len(step_times[-50:])
            lr_now = optimizer.param_groups[0]["lr"]
            # track weight change from init
            w_now = policy.model.action_in_proj.weight.data
            delta = (w_now - w0).abs().max().item()
            eta = (args.train_steps - step) * avg_t
            log.info(
                f"step {step:5d}/{args.train_steps}  "
                f"loss={loss.item():.5f}  L_FM={loss_dict['loss_fm'].item():.5f}  "
                f"lr={lr_now:.2e}  grad_norm={grad_norm:.3f}  "
                f"Δai_proj={delta:.5f}  step_t={avg_t:.2f}s  ETA={eta/3600:.1f}h"
            )

        if (step + 1) % args.save_every == 0:
            ckpt = out_path.parent / f"stage8b_step{step+1:05d}.safetensors"
            _save(policy, ckpt)
            log.info(f"Checkpoint: {ckpt}")

    _save(policy, out_path)
    log.info(f"Final: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path",       default=PRETRAINED_PATH)
    parser.add_argument("--warmstart_ckpt",         default=STUDENT_CKPT)
    parser.add_argument("--normalizer_stats_path", default=NORMALIZER_STATS_PATH)
    parser.add_argument("--dataset_path",          default=DATASET_PATH)
    parser.add_argument("--output_path",           default=OUTPUT_PATH)
    parser.add_argument("--train_steps", type=int, default=5_000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--grad_clip",   type=float, default=1.0)
    parser.add_argument("--save_every",  type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--local_gpu",   type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
