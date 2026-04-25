"""SnapFlow distillation trainer — 논문 방식 그대로.

개선사항:
  - VLM backbone 완전 freeze (action expert + target_time_emb만 학습)
  - TargetTimeEmbedding 추가 (zero-init, s 인코딩)
  - 혼합 손실: L = α·L_FM + (1-α)·λ·L_shortcut
  - 외부 teacher 제거 (self-distillation)

Usage:
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
        scripts/stage1_snapflow_distill.py \
        --dataset_path /data/jameskimh/james_libero_datasets/libero_10 \
        --pretrained_path /data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned \
        --output_path artifacts/stage1_student.safetensors \
        --train_steps 30000
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

PRETRAINED_PATH = "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
NORMALIZER_STATS_PATH = (
    "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
    "/policy_preprocessor_step_2_normalizer_processor.safetensors"
)
DATASET_PATH = "/data/jameskimh/james_libero_datasets/libero_10"
OUTPUT_PATH = "artifacts/stage1_student.safetensors"

# 논문 하이퍼파라미터 (Table 5 ablation: α=0.5, λ=0.1이 최적)
ALPHA = 0.5   # FM vs shortcut 비율
LAM   = 0.1   # shortcut 가중치


def _setup_paths():
    root = Path(__file__).resolve().parents[2]
    for p in [
        str(root / "Snapflow_QuaRot"),
        str(root / "lerobot" / "src"),
        str(root / "TensorRT-Model-Optimizer"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_ddp() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_ddp() else 1


def _load_policy(pretrained_path: str, device: torch.device):
    import torch._dynamo as _dynamo
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy
    from lerobot.envs.configs import LiberoEnv

    _dynamo.reset()
    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    policy_cfg.pretrained_path = pretrained_path
    policy_cfg.device = str(device)
    policy_cfg.use_amp = False
    policy_cfg.n_action_steps = 10
    policy_cfg.num_inference_steps = 10

    env_cfg = LiberoEnv(task="libero_10", task_ids=list(range(10)))
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    return policy.to(device)


def _patch_embed_image(policy):
    """transformers 4.57.6에서 get_image_features API 변경 대응.

    이전 API: get_image_features() → object with .pooler_output
    신규 API: get_image_features() → projected_tensor / sqrt(hidden_size)

    embed_image가 직접 Python 코드로 실행될 때만 문제 발생.
    (eval에서는 torch.compile이 캐시된 그래프를 쓰기 때문에 우회됨)
    """
    pali_with_expert = policy.model.paligemma_with_expert

    def patched_embed_image(image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        pali = pali_with_expert.paligemma
        # get_image_features 반환값 처리: 구 버전(object) vs 신 버전(tensor)
        image_outputs = pali.model.get_image_features(image)
        if hasattr(image_outputs, "pooler_output"):
            # 구 API: pooler_output 존재
            features = image_outputs.pooler_output * pali.config.text_config.hidden_size ** 0.5
        else:
            # 신 API(4.57.6): get_image_features = multi_modal_projector(last_hs) / sqrt(hidden)
            # 다시 sqrt(hidden) 곱해서 projected features 복원
            features = image_outputs * pali.config.text_config.hidden_size ** 0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    pali_with_expert.embed_image = patched_embed_image
    log.info("embed_image patched for transformers API compatibility")


def _freeze_vlm_unfreeze_expert(policy):
    """VLM backbone freeze, action expert + φs만 학습.

    논문: "freezes the VLM backbone and trains only the action expert
    and φs — about 10% of parameters"
    """
    # 전체 freeze
    for p in policy.parameters():
        p.requires_grad_(False)

    inner = policy.model
    # action expert (Gemma expert + action proj + time MLP)
    modules_to_train = [
        inner.paligemma_with_expert.gemma_expert,
        inner.action_in_proj,
        inner.action_out_proj,
        inner.time_mlp_in,
        inner.time_mlp_out,
    ]
    for m in modules_to_train:
        for p in m.parameters():
            p.requires_grad_(True)

    n_total = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    pct = 100.0 * n_train / n_total
    log.info(f"VLM frozen. Trainable: {n_train:,} / {n_total:,} ({pct:.1f}%)")


def train(
    pretrained_path: str = PRETRAINED_PATH,
    normalizer_stats_path: str = NORMALIZER_STATS_PATH,
    dataset_path: str = DATASET_PATH,
    output_path: str = OUTPUT_PATH,
    train_steps: int = 30_000,
    batch_size: int = 4,
    lr: float = 1e-5,
    grad_clip: float = 1.0,
    save_every: int = 5000,
    num_workers: int = 4,
    seed: int = 42,
    alpha: float = ALPHA,
    lam: float = LAM,
    warmstart_ckpt: str | None = None,
):
    _setup_paths()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    use_ddp = "LOCAL_RANK" in os.environ and torch.cuda.device_count() > 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        if _rank() == 0:
            log.info(f"DDP: {_world_size()} processes")

    from snapflow.data import LiberoHDF5Dataset
    from snapflow.distill_loss import SnapFlowLoss
    from snapflow.target_time_emb import TargetTimeEmbedding
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    # ── 정책 로드 ──────────────────────────────────────────────────────────
    if _rank() == 0:
        log.info("Loading policy (self-distillation: same model for FM + shortcut)...")
    policy = _load_policy(pretrained_path, device)

    # transformers API 호환성 패치 (4.57.6에서 get_image_features 반환 타입 변경)
    _patch_embed_image(policy)

    # Optional: warmstart from prior checkpoint (e.g., FM-only stage1_student.safetensors)
    if warmstart_ckpt is not None:
        from safetensors.torch import load_file as _load_file
        ckpt_state = _load_file(str(warmstart_ckpt))
        policy_state = {k: v for k, v in ckpt_state.items() if not k.startswith("target_time_emb.")}
        missing, unexpected = policy.load_state_dict(policy_state, strict=False)
        if _rank() == 0:
            log.info(f"Warmstart from {warmstart_ckpt}: missing={len(missing)}, unexpected={len(unexpected)}")

    # VLM freeze, action expert만 학습
    _freeze_vlm_unfreeze_expert(policy)

    # V4: target-time embedding φs → action output space (max_action_dim=32)
    # V3 실패 원인: embed_dim=2048으로 adarms_cond에 주입 → AdaRMS 오염
    # V4 fix: out_dim=max_action_dim (32)으로 action velocity에 주입
    action_out_dim = policy.model.action_out_proj.out_features  # max_action_dim (32)
    target_time_emb = TargetTimeEmbedding(out_dim=action_out_dim, sinusoidal_dim=256, hidden_dim=256).to(device)
    if _rank() == 0:
        log.info(f"TargetTimeEmbedding: out_dim={action_out_dim} (action space), params={sum(p.numel() for p in target_time_emb.parameters()):,}")

    # DDP 래핑
    if use_ddp:
        policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=True)
        target_time_emb = DDP(target_time_emb, device_ids=[local_rank])

    policy_core = policy.module if use_ddp else policy
    tte_core = target_time_emb.module if use_ddp else target_time_emb

    # ── 데이터 ────────────────────────────────────────────────────────────
    dataset = LiberoHDF5Dataset(
        dataset_path=dataset_path,
        normalizer_stats_path=normalizer_stats_path,
        seed=seed,
    )
    if use_ddp:
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, drop_last=True, pin_memory=True,
                            persistent_workers=(num_workers > 0))
    else:
        g = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True, pin_memory=True,
                            generator=g, persistent_workers=(num_workers > 0))

    # ── 옵티마이저 ────────────────────────────────────────────────────────
    # phi_s(TTE)는 action expert보다 lr을 10x 낮게: 빠른 성장이 adarms_cond를 오염시키는 것을 방지
    action_expert_params = [p for p in policy_core.parameters() if p.requires_grad]
    tte_params = list(target_time_emb.parameters())
    trainable_params = action_expert_params + tte_params
    optimizer = AdamW(
        [
            {"params": action_expert_params, "lr": lr},
            {"params": tte_params, "lr": lr * 0.1},   # phi_s: 10x 낮은 lr
        ],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_steps, eta_min=lr * 0.1)

    loss_fn = SnapFlowLoss(action_dim=7, alpha=alpha, lam=lam)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _rank() == 0:
        log.info(f"Training: {train_steps} steps, batch={batch_size}, lr={lr}, alpha={alpha}, lam={lam}")
        log.info(f"Dataset: {len(dataset)} samples")

    loader_iter = iter(loader)
    step_times = []

    for step in range(train_steps):
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
            loss_dict = loss_fn(policy_core, batch, target_time_emb=target_time_emb)

        loss = loss_dict["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()
        scheduler.step()

        elapsed = time.perf_counter() - t0
        step_times.append(elapsed)

        if _rank() == 0 and step % 50 == 0:
            avg_t = sum(step_times[-50:]) / len(step_times[-50:])
            remaining = (train_steps - step) * avg_t
            # phi_s norm 모니터링 (s=0, 0.5, 1.0 각각)
            with torch.no_grad():
                s_vals = torch.tensor([0.0, 0.5, 1.0], device=device)
                phi_norms = [tte_core(s_vals[i:i+1]).norm().item() for i in range(3)]
            log.info(
                f"step {step:5d}/{train_steps}  "
                f"loss={loss.item():.5f}  "
                f"L_FM={loss_dict['loss_fm'].item():.5f}  "
                f"L_sc={loss_dict['loss_shortcut'].item():.5f}  "
                f"phi_s(0/0.5/1)={phi_norms[0]:.3f}/{phi_norms[1]:.3f}/{phi_norms[2]:.3f}  "
                f"step_t={avg_t:.2f}s  ETA={remaining/3600:.1f}h"
            )

        if _rank() == 0 and (step + 1) % save_every == 0:
            ckpt = out_path.parent / f"step{step+1:05d}.safetensors"
            _save_checkpoint(policy_core, tte_core, ckpt)
            log.info(f"Checkpoint: {ckpt}")

    if _rank() == 0:
        _save_checkpoint(policy_core, tte_core, out_path)
        log.info(f"Final student checkpoint: {out_path}")

    if use_ddp:
        dist.destroy_process_group()


def _save_checkpoint(policy, target_time_emb, path: Path):
    """policy + target_time_emb 가중치를 하나의 safetensors로 저장."""
    from safetensors.torch import save_file
    state = {}
    for k, v in policy.state_dict().items():
        state[k] = v.contiguous().cpu()
    # target_time_emb 가중치를 "target_time_emb." prefix로 저장
    for k, v in target_time_emb.state_dict().items():
        state[f"target_time_emb.{k}"] = v.contiguous().cpu()
    save_file(state, str(path))
