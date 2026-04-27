"""Stage 8 — SnapFlow distillation on INT4 g=16 quantized model.

기존 stage1은 FP16 teacher에서 distillation → NFE=1 student가 FP16 LLM features에 맞게 학습.
Stage 7 결과(45%): INT4 g=16 quantized model에 stage1 student를 그대로 쓰면 NFE=1에서 성능 저하.

해결: INT4 g=16 quantized model을 teacher로 삼아 직접 distillation.
 - R1+R3 rotation + INT4 g=16 fake-quant 모델에서 FM velocity를 teacher signal로 사용
 - warmstart: stage1_student.safetensors (FP16-trained student → quant model에 적응)
 - FM-only loss (shortcut loss 없이, time_mlp corruption 방지)
 - freeze time_mlp (V5 fix)

Run (2 GPUs):
    CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 torchrun --nproc_per_node=2 \\
        scripts/stage8_snapflow_qat_distill.py \\
        --train_steps 10000 --batch_size 4 --lr 1e-5
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

PRETRAINED_PATH = "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
STUDENT_CKPT    = "artifacts/stage1_student.safetensors"
NORMALIZER_STATS_PATH = (
    "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
    "/policy_preprocessor_step_2_normalizer_processor.safetensors"
)
DATASET_PATH    = "/data/jameskimh/james_libero_datasets/libero_10"
OUTPUT_PATH     = "artifacts/stage8_student.safetensors"
OHB_MANIFEST_PATH = "artifacts/stage4_ohb_manifest.json"
_R3_SEED        = 100


def _rank() -> int:
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

def _world() -> int:
    return dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1


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
    cfg.gradient_checkpointing = False  # pretrained config has True → disable for training speed
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
    if _rank() == 0:
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
        if _rank() == 0:
            log.info(f"Calibration: {n} batches")

    mtq.quantize(policy, quant_cfg, forward_loop=forward_loop)
    if _rank() == 0:
        log.info(f"INT4 g=16 applied (OHB: {len(ohb_manifest)} protected)")


def _freeze_vlm_unfreeze_expert(policy):
    """VLM + quantized layers 모두 freeze, action expert만 학습."""
    for p in policy.parameters():
        p.requires_grad_(False)

    inner = policy.model
    for m in [
        inner.paligemma_with_expert.gemma_expert,
        inner.action_in_proj,
        inner.action_out_proj,
        # time_mlp frozen (V5 fix: adarms_cond protection)
    ]:
        for p in m.parameters():
            p.requires_grad_(True)

    n_total = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if _rank() == 0:
        log.info(f"Frozen. Trainable: {n_train:,}/{n_total:,} ({100.*n_train/n_total:.1f}%)")


def train(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    use_ddp = "LOCAL_RANK" in os.environ and torch.cuda.device_count() > 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        if _rank() == 0:
            log.info(f"DDP: {_world()} GPUs")

    # ── 모델 로드 ──────────────────────────────────────────────────────────
    if _rank() == 0:
        log.info("Loading policy...")
    policy = _load_policy(args.pretrained_path, device)
    _patch_embed_image(policy)

    # ── Warmstart from stage1 student ────────────────────────────────────
    if Path(args.warmstart_ckpt).exists():
        from safetensors.torch import load_file
        state = load_file(args.warmstart_ckpt)
        # target_time_emb 키 제외 (구조 다를 수 있음)
        state = {k: v for k, v in state.items() if not k.startswith("target_time_emb.")}
        missing, unexpected = policy.load_state_dict(state, strict=False)
        if _rank() == 0:
            log.info(f"Warmstart: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        if _rank() == 0:
            log.warning(f"Warmstart ckpt not found: {args.warmstart_ckpt}, training from scratch")

    # ── Rotation + Quantization (각 rank 독립적으로, 동일 시드 → 동일 결과) ──
    if _rank() == 0:
        log.info("Applying R1+R3 rotation...")
    _apply_quarot(policy, device)

    if _rank() == 0:
        log.info("Applying INT4 g=16 (fake quant)...")
    _apply_int4_g16(policy, device)

    # ── Post-quant resume: rotation이 이미 적용된 checkpoint에서 expert만 복원 ──
    # stage8_step* checkpoint는 이미 R3-rotated + QAT-trained expert weights를 포함.
    # rotation 이후에 로드해야 이중 회전 문제를 방지할 수 있다.
    if getattr(args, "resume_after_quant_ckpt", None):
        resume_path = Path(args.resume_after_quant_ckpt)
        if resume_path.exists():
            from safetensors.torch import load_file
            resume_state = load_file(str(resume_path))
            _RESUME_PREFIXES = (
                "model.paligemma_with_expert.gemma_expert.",
                "model.action_in_proj.",
                "model.action_out_proj.",
            )
            expert_state = {
                k: v for k, v in resume_state.items()
                if any(k.startswith(p) for p in _RESUME_PREFIXES)
            }
            missing, unexpected = policy.load_state_dict(expert_state, strict=False)
            if _rank() == 0:
                log.info(
                    f"Post-quant resume from {resume_path.name}: "
                    f"{len(expert_state)} expert keys loaded "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )
        elif _rank() == 0:
            log.warning(f"resume_after_quant_ckpt not found: {resume_path}")

    # ── Freeze VLM, unfreeze action expert ───────────────────────────────
    _freeze_vlm_unfreeze_expert(policy)

    # ── DDP 래핑 ─────────────────────────────────────────────────────────
    if use_ddp:
        policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=True)
    policy_core = policy.module if use_ddp else policy

    # ── 데이터 ────────────────────────────────────────────────────────────
    from snapflow.data import LiberoHDF5Dataset
    from snapflow.distill_loss import SnapFlowLoss
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    dataset = LiberoHDF5Dataset(
        dataset_path=args.dataset_path,
        normalizer_stats_path=args.normalizer_stats_path,
        seed=args.seed,
    )
    if use_ddp:
        sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.num_workers, drop_last=True, pin_memory=True,
                            persistent_workers=False)
    else:
        g = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True, pin_memory=True,
                            generator=g, persistent_workers=(args.num_workers > 0))

    # ── 옵티마이저 ────────────────────────────────────────────────────────
    trainable_params = [p for p in policy_core.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.train_steps, eta_min=args.lr * 0.1)

    # FM-only loss (alpha=1.0, lam=0 → shortcut loss 없음)
    loss_fn = SnapFlowLoss(action_dim=7, alpha=1.0, lam=0.0)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _rank() == 0:
        log.info(f"Training {args.train_steps} steps, batch={args.batch_size}×{_world()}, lr={args.lr}")
        log.info(f"Dataset: {len(dataset)} samples")

    loader_iter = iter(loader)
    step_times = []

    for step in range(args.train_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            if use_ddp:
                sampler.set_epoch(step)
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
                 for k, v in batch.items()}

        t0 = time.perf_counter()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss_dict = loss_fn(policy_core, batch)

        loss = loss_dict["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        step_times.append(time.perf_counter() - t0)

        if _rank() == 0 and step % 100 == 0:
            avg_t = sum(step_times[-50:]) / len(step_times[-50:])
            eta   = (args.train_steps - step) * avg_t
            log.info(
                f"step {step:5d}/{args.train_steps}  "
                f"loss={loss.item():.5f}  "
                f"L_FM={loss_dict['loss_fm'].item():.5f}  "
                f"step_t={avg_t:.2f}s  ETA={eta/3600:.1f}h"
            )

        if _rank() == 0 and (step + 1) % args.save_every == 0:
            ckpt = out_path.parent / f"stage8_step{step+1:05d}.safetensors"
            _save(policy_core, ckpt)
            log.info(f"Checkpoint: {ckpt}")

    if _rank() == 0:
        _save(policy_core, out_path)
        log.info(f"Final: {out_path}")

    if use_ddp:
        dist.destroy_process_group()


def _save(policy, path: Path):
    from safetensors.torch import save_file
    state = {k: v.contiguous().cpu() for k, v in policy.state_dict().items()}
    save_file(state, str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path",        default=PRETRAINED_PATH)
    parser.add_argument("--warmstart_ckpt",          default=STUDENT_CKPT)
    parser.add_argument("--resume_after_quant_ckpt", default=None,
                        help="rotation+quant 이후에 expert weights를 로드할 checkpoint "
                             "(stage8_step*.safetensors 등). 이중 회전 방지용.")
    parser.add_argument("--normalizer_stats_path",  default=NORMALIZER_STATS_PATH)
    parser.add_argument("--dataset_path",           default=DATASET_PATH)
    parser.add_argument("--output_path",            default=OUTPUT_PATH)
    parser.add_argument("--train_steps",  type=int, default=10_000)
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--grad_clip",    type=float, default=1.0)
    parser.add_argument("--save_every",   type=int, default=2000)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
