"""Precompute FP16 teacher velocity targets and save to disk.

Teacher (Path A, FP16 frozen)의 2-step Euler velocity target을 미리 계산해 저장.
이후 stage8f_qad.py에서 --teacher_cache로 로드하면:
  - Teacher 모델을 매 QAD 실험마다 다시 로드할 필요 없음
  - GPU 메모리 절약 (~30 GB)
  - Training step당 teacher forward 2회 생략 → 속도 향상

Usage:
  python precompute_teacher_cache.py \\
      --n_batches 512 --batch_size 4 --seed 42 \\
      --output /tmp/teacher_cache.pt
"""
import argparse
import sys
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
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.policies.factory import make_policy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

from snapflow.data import LiberoHDF5Dataset
from snapflow.distill_loss import get_velocity

PRETRAINED_PATH = "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
PATH_A          = "/tmp/path_a.pt"
CALIB_PATH      = "/data/jameskimh/james_libero_datasets/libero_10"
NORMALIZER_PATH = (
    "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
    "/policy_preprocessor_step_2_normalizer_processor.safetensors"
)
DEVICE = "cuda"


def _patch_embed_image(policy):
    pw = policy.model.paligemma_with_expert
    def pe(image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        p = pw.paligemma
        out = p.model.get_image_features(image)
        f = (out.pooler_output if hasattr(out, "pooler_output") else out) \
            * p.config.text_config.hidden_size ** 0.5
        return f.to(out_dtype)
    pw.embed_image = pe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_batches", type=int, default=512,
                        help="Number of batches to precompute (covers ~512 QAD steps)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--output", default="/tmp/teacher_cache.pt",
                        help="Output path for the cache file")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load teacher ──────────────────────────────────────────────────────────
    print("[INFO] Loading FP16 teacher (Path A)...")
    policy_cfg = PreTrainedConfig.from_pretrained(PRETRAINED_PATH)
    policy_cfg.pretrained_path  = PRETRAINED_PATH
    policy_cfg.device           = args.device
    policy_cfg.use_amp          = False
    policy_cfg.compile_model    = False
    policy_cfg.gradient_checkpointing = False
    policy_cfg.n_action_steps   = 10
    policy_cfg.num_inference_steps = 1

    # Minimal env config (only needed for make_policy shape info)
    env_cfg = LiberoEnv(task="libero_10", task_ids=list(range(10)))
    teacher = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    _patch_embed_image(teacher)

    sd = torch.load(PATH_A, map_location="cpu")
    teacher.load_state_dict(sd, strict=False)
    teacher.to(args.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("[INFO] Teacher loaded.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = LiberoHDF5Dataset(CALIB_PATH, NORMALIZER_PATH, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))

    # ── Precompute v_target ───────────────────────────────────────────────────
    cache = []
    data_iter = iter(loader)
    print(f"[INFO] Precomputing {args.n_batches} batches...")

    with torch.no_grad():
        for i in range(args.n_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            batch_gpu = {k: v.to(args.device) if hasattr(v, "to") else v
                         for k, v in batch.items()}

            images, img_masks = teacher._preprocess_images(batch_gpu)
            tokens  = batch_gpu[OBS_LANGUAGE_TOKENS]
            masks   = batch_gpu[OBS_LANGUAGE_ATTENTION_MASK]
            actions = teacher.prepare_action(batch_gpu)
            B = actions.shape[0]

            # Fixed noise per batch for reproducibility
            torch.manual_seed(args.seed + i)
            noise = teacher.model.sample_noise(actions.shape, args.device)

            t1  = torch.ones(B, device=args.device, dtype=actions.dtype)
            t05 = torch.full((B,), 0.5, device=args.device, dtype=actions.dtype)

            # 2-step Euler teacher reference
            v1  = get_velocity(teacher.model, images, img_masks, tokens, masks,
                               noise, t1, s=t1)
            x05 = noise - 0.5 * v1
            v05 = get_velocity(teacher.model, images, img_masks, tokens, masks,
                               x05, t05, s=t05)
            v_target = 0.5 * (v1 + v05)  # trapezoidal avg

            # Save CPU tensors to avoid GPU memory bloat
            # images / img_masks may be list-of-tensors (multi-camera)
            def _cpu_half(x):
                if isinstance(x, (list, tuple)):
                    return [t.cpu().half() if t.is_floating_point() else t.cpu() for t in x]
                return x.cpu().half() if x.is_floating_point() else x.cpu()

            def _cpu(x):
                if isinstance(x, (list, tuple)):
                    return [t.cpu() for t in x]
                return x.cpu()

            cache.append({
                "images":    _cpu_half(images),
                "img_masks": _cpu(img_masks),
                "tokens":    tokens.cpu(),
                "masks":     masks.cpu(),
                "noise":     noise.cpu().half(),
                "v_target":  v_target.cpu().half(),
            })

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{args.n_batches}] batch done")

    print(f"[INFO] Saving cache → {out_path}")
    torch.save({
        "n_batches":  len(cache),
        "batch_size": args.batch_size,
        "seed":       args.seed,
        "batches":    cache,
    }, out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f"[INFO] Cache saved: {len(cache)} batches, {size_mb:.1f} MB")
    print("[INFO] Done. Use with: --teacher_cache", out_path)


if __name__ == "__main__":
    main()
