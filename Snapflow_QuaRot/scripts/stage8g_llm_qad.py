"""Stage 8g — Phase 1: LLM Quantization-Aware Distillation.

Sequential LLM→DiT QAD의 Phase 1.

  - LLM (PaliGemma: vision_tower + language_model + projector) W4A4 fake-quant
  - DiT (GemmaExpert) FP16 frozen — velocity 계산에만 사용
  - Loss: MSE(v_student_NFE1, v_teacher) — stage8f_qad.py와 동일 loss
  - 출력: best_llm_student.pt  (W4A4 LLM trained + FP16 DiT weights)

Phase 2 진행 방법:
  python stage8f_qad.py --quant_format w4a4 --group_size 8 \\
      --llm_ckpt results/stage8g_llm_qad_g8/best_llm_student.pt \\
      --teacher_cache /tmp/teacher_cache.pt

Usage:
  python stage8g_llm_qad.py --group_size 8 --n_steps 500 --lr 1e-4 \\
      --teacher_cache /tmp/teacher_cache.pt
"""
import argparse
import copy
import json
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
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


# ── quantization config ──────────────────────────────────────────────────────

def _build_w4a4_llm_config(group_size: int = 8, scale_bits: int = None) -> dict:
    """LLM (PaliGemma) 만 W4A4. DiT (GemmaExpert) 는 FP16 유지."""
    block_sizes = {-1: group_size}
    if scale_bits is not None:
        block_sizes["scale_bits"] = scale_bits
    w4   = {"num_bits": 4, "block_sizes": block_sizes, "enable": True}
    a4   = {"num_bits": 4, "enable": True}
    fp16 = {"enable": False}
    return {
        "quant_cfg": {
            # LLM components → W4A4
            "*language_model*weight_quantizer":        w4,
            "*vision_tower*weight_quantizer":          w4,
            "*multi_modal_projector*weight_quantizer": w4,
            "*language_model*input_quantizer":         a4,
            "*vision_tower*input_quantizer":           a4,
            "*multi_modal_projector*input_quantizer":  a4,
            # DiT + special layers → FP16
            "*gemma_expert*":      fp16,
            "*action_in_proj*":    fp16,
            "*action_out_proj*":   fp16,
            "*lm_head*":           fp16,
            "*[kv]_bmm_quantizer": fp16,
            "default":             fp16,
        },
        "algorithm": "max",
    }


def _apply_llm_fake_quant(policy, group_size: int, device: str,
                          scale_bits: int = None):
    import modelopt.torch.quantization as mtq

    cfg = _build_w4a4_llm_config(group_size, scale_bits=scale_bits)
    scale_label = f"INT{scale_bits}" if scale_bits else "FP16"
    print(f"[INFO] Applying LLM fake-quant: W4A4 g={group_size}, scale={scale_label}")

    def calib_loop(model):
        model.eval()
        dataset = LiberoHDF5Dataset(CALIB_PATH, NORMALIZER_PATH, seed=0)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        n = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if hasattr(v, "to") else v
                         for k, v in batch.items()}
                try:
                    images, img_masks = model._preprocess_images(batch)
                    tokens  = batch[OBS_LANGUAGE_TOKENS]
                    masks   = batch[OBS_LANGUAGE_ATTENTION_MASK]
                    actions = model.prepare_action(batch)
                    noise   = model.model.sample_noise(actions.shape, device)
                    t       = torch.ones(actions.shape[0], device=device)
                    get_velocity(model.model, images, img_masks,
                                 tokens, masks, noise, t, s=t)
                except Exception:
                    pass
                n += 1
                if n >= 32:
                    break
        print(f"[INFO] Calibration done: {n} batches")

    mtq.quantize(policy, cfg, forward_loop=calib_loop)

    # Verify / patch invalid amax
    bad, ok = [], []
    for name, mod in policy.named_modules():
        for qname in ("input_quantizer", "weight_quantizer"):
            q = getattr(mod, qname, None)
            if q is None or not getattr(q, "_enabled", False):
                continue
            amax = getattr(q, "_amax", None)
            if amax is None or (isinstance(amax, torch.Tensor) and (amax < 0).any()):
                bad.append(f"{name}.{qname}")
            else:
                ok.append(f"{name}.{qname}")
    print(f"[INFO] Quantizer amax check: {len(ok)} valid, {len(bad)} invalid")
    if bad:
        for name, mod in policy.named_modules():
            for qname in ("input_quantizer", "weight_quantizer"):
                q = getattr(mod, qname, None)
                if q is None:
                    continue
                amax = getattr(q, "_amax", None)
                if amax is None or (isinstance(amax, torch.Tensor) and (amax < 0).any()):
                    q._amax = torch.ones(1, device=next(policy.parameters()).device)
        print("[INFO] Patched invalid amax → 1.0")

    print("[INFO] LLM fake-quant registered (W4A4)")


# ── model utils ──────────────────────────────────────────────────────────────

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


def _freeze_for_llm_qad(policy):
    """DiT + lm_head freeze, LLM (language_model/vision_tower/projector) 학습.

    Phase 1 목표: W4A4 LLM이 FP16 LLM과 동일한 velocity를 생성하도록 학습.
    DiT는 FP16으로 frozen — velocity 계산에만 활용.
    """
    TRAIN_PATTERNS = [
        "language_model",
        "vision_tower",
        "multi_modal_projector",
    ]
    ALWAYS_FREEZE = [
        "lm_head",  # vocab 투영 — robot control 경로에서 불필요
    ]
    n_frozen = n_trainable = 0
    for name, param in policy.named_parameters():
        if any(p in name for p in ALWAYS_FREEZE):
            param.requires_grad_(False)
            n_frozen += param.numel()
        elif any(p in name for p in TRAIN_PATTERNS):
            param.requires_grad_(True)
            n_trainable += param.numel()
        else:
            param.requires_grad_(False)
            n_frozen += param.numel()
    print(f"[INFO] LLM QAD — Trainable (LLM): {n_trainable/1e6:.1f}M  "
          f"Frozen (DiT+lm_head): {n_frozen/1e6:.1f}M")


def _reset_dynamo(policy):
    import torch._dynamo as _dynamo
    _dynamo.reset()
    inner = getattr(policy, "model", None)
    if inner:
        for attr in ("sample_actions", "forward"):
            fn = getattr(inner, attr, None)
            if fn:
                orig = (getattr(fn, "_torchdynamo_orig_callable", None)
                        or getattr(fn, "_orig_mod", None))
                if orig:
                    setattr(inner, attr, orig)


# ── training loop ────────────────────────────────────────────────────────────

def _load_teacher_cache(cache_path: str, device: str):
    print(f"[INFO] Loading teacher cache from {cache_path}")
    data = torch.load(cache_path, map_location="cpu")
    batches = data["batches"]
    print(f"[INFO] Cache loaded: {len(batches)} batches, batch_size={data['batch_size']}")
    return batches


def _to_device(x, device, dtype=None):
    """Move tensor or list-of-tensors to device, optionally casting dtype."""
    if isinstance(x, (list, tuple)):
        return [t.to(device, dtype=dtype) if dtype and t.is_floating_point() else t.to(device)
                for t in x]
    return x.to(device, dtype=dtype) if dtype and x.is_floating_point() else x.to(device)


def _llm_qad_train(student, teacher, device: str, n_steps: int, lr: float,
                   batch_size: int, out_dir: Path, action_dim: int = 7,
                   teacher_cache: str = None):
    """
    Phase 1 학습 루프.

    Teacher (FP16 full model): v_target 생성
    Student (W4A4 LLM + FP16 DiT): v_pred = DiT(W4A4_LLM(obs))
    Loss: MSE(v_pred, v_target)
    Update: LLM parameters only
    """
    if teacher_cache:
        cached_batches = _load_teacher_cache(teacher_cache, device)
        cache_iter = iter(cached_batches * ((n_steps // len(cached_batches)) + 2))
        use_cache = True
    else:
        dataset = LiberoHDF5Dataset(CALIB_PATH, NORMALIZER_PATH, seed=42)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, drop_last=True, pin_memory=True)
        data_iter = iter(loader)
        use_cache = False

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    student.train()
    if teacher is not None:
        teacher.eval()

    log_rows = []
    best_loss = float("inf")
    best_ckpt = out_dir / "best_llm_student.pt"

    for step in range(1, n_steps + 1):
        if use_cache:
            entry     = next(cache_iter)
            images    = _to_device(entry["images"],    device, dtype=torch.float32)
            img_masks = _to_device(entry["img_masks"], device)
            tokens    = entry["tokens"].to(device)
            masks     = entry["masks"].to(device)
            noise     = entry["noise"].to(device, dtype=torch.float32)
            v_target  = entry["v_target"].to(device, dtype=torch.float32)
            B  = noise.shape[0]
            t1 = torch.ones(B, device=device)
        else:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            batch = {k: v.to(device) if hasattr(v, "to") else v
                     for k, v in batch.items()}
            images, img_masks = student._preprocess_images(batch)
            tokens  = batch[OBS_LANGUAGE_TOKENS]
            masks   = batch[OBS_LANGUAGE_ATTENTION_MASK]
            actions = student.prepare_action(batch)
            B       = actions.shape[0]
            noise   = student.model.sample_noise(actions.shape, device)
            t1      = torch.ones(B, device=device, dtype=actions.dtype)
            t05     = torch.full((B,), 0.5, device=device, dtype=actions.dtype)

            with torch.no_grad():
                v1  = get_velocity(teacher.model, images, img_masks,
                                   tokens, masks, noise, t1, s=t1)
                x05 = noise - 0.5 * v1
                v05 = get_velocity(teacher.model, images, img_masks,
                                   x05, t05, s=t05)
                v_target = 0.5 * (v1 + v05)

        # Student forward: W4A4 LLM → FP16 DiT → v_pred
        s0     = torch.zeros(B, device=device)
        v_pred = get_velocity(student.model, images, img_masks,
                              tokens, masks, noise, t1, s=s0)

        loss = F.mse_loss(v_pred[:, :, :action_dim], v_target[:, :, :action_dim])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        if step % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"[step {step:4d}/{n_steps}] loss={loss_val:.6f}  lr={lr_now:.2e}")
            log_rows.append({"step": step, "loss": loss_val, "lr": lr_now})

        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(student.state_dict(), best_ckpt)

    torch.save(student.state_dict(), out_dir / "final_llm_student.pt")
    print(f"[INFO] Best loss={best_loss:.6f}  saved → {best_ckpt}")
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(log_rows, f, indent=2)

    return best_ckpt


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--scale_bits", type=int, default=None,
                        help="INT16 scale (None=FP16)")
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=10)
    parser.add_argument("--task_ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--teacher_cache", default=None,
                        help="precompute_teacher_cache.py 출력 경로. "
                             "제공시 teacher 모델 로드 생략 → ~30 GB 절약.")
    args = parser.parse_args()

    scale_suffix = f"_s{args.scale_bits}" if args.scale_bits else ""
    out_dir = Path(args.output_dir or
                   f"results/stage8g_llm_qad_g{args.group_size}{scale_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load base policy ──────────────────────────────────────────────────
    policy_cfg = PreTrainedConfig.from_pretrained(PRETRAINED_PATH)
    policy_cfg.pretrained_path  = PRETRAINED_PATH
    policy_cfg.device           = args.device
    policy_cfg.use_amp          = False
    policy_cfg.compile_model    = False
    policy_cfg.gradient_checkpointing = False
    policy_cfg.n_action_steps   = 10
    policy_cfg.num_inference_steps = 1

    print("[INFO] Building student policy (Phase 1: LLM QAD) ...")
    env_cfg    = LiberoEnv(task="libero_10", task_ids=args.task_ids)
    envs_dict  = make_env(env_cfg, n_envs=args.eval_batch_size)

    student = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    _patch_embed_image(student)

    sd = torch.load(PATH_A, map_location="cpu")
    student.load_state_dict(sd, strict=False)
    print("[INFO] Path A loaded into student")

    if not args.skip_train:
        # ── teacher ───────────────────────────────────────────────────────
        if args.teacher_cache:
            print(f"[INFO] Using precomputed teacher cache: {args.teacher_cache}")
            teacher = None
        else:
            print("[INFO] Building FP16 teacher ...")
            teacher = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
            _patch_embed_image(teacher)
            teacher.load_state_dict(sd, strict=False)
            teacher.to(args.device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

        # ── apply LLM W4A4 fake-quant ─────────────────────────────────────
        _apply_llm_fake_quant(student, args.group_size, args.device,
                              scale_bits=args.scale_bits)
        _freeze_for_llm_qad(student)
        _reset_dynamo(student)
        student.to(args.device)

        print(f"\n[INFO] Phase 1 LLM QAD: W4A4 g={args.group_size} | "
              f"steps={args.n_steps} | lr={args.lr}")
        best_ckpt = _llm_qad_train(
            student=student,
            teacher=teacher,
            device=args.device,
            n_steps=args.n_steps,
            lr=args.lr,
            batch_size=args.train_batch_size,
            out_dir=out_dir,
            teacher_cache=args.teacher_cache,
        )

        if teacher is not None:
            del teacher
        torch.cuda.empty_cache()

    else:
        best_ckpt = out_dir / "best_llm_student.pt"
        print(f"[INFO] skip_train: loading {best_ckpt}")
        _apply_llm_fake_quant(student, args.group_size, args.device,
                              scale_bits=args.scale_bits)
        best_sd = torch.load(best_ckpt, map_location="cpu")
        student.load_state_dict(best_sd, strict=False)

    # ── eval ──────────────────────────────────────────────────────────────
    student.eval()
    _reset_dynamo(student)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=PRETRAINED_PATH,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "tokenizer_processor": {"tokenizer_name": "/tmp/paligemma_tok_fast"},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    print(f"\n[INFO] Evaluating Phase 1 (W4A4 LLM + FP16 DiT) ...")
    with torch.no_grad():
        eval_info = eval_policy_all(
            envs=envs_dict,
            policy=student,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.n_episodes,
            start_seed=args.start_seed,
        )

    for suite_envs in envs_dict.values():
        for env in suite_envs.values():
            try: env.close()
            except Exception: pass

    overall = eval_info.get("overall", {})
    pc      = overall.get("pc_success", float("nan"))

    print(f"\n{'='*60}")
    print(f"[Phase 1 LLM QAD g={args.group_size}] pc_success={pc:.1f}%")
    print(f"  Reference: FP16 (Path A) = 100.0%")
    print(f"  W4A16 llm_only PTQ       =  78.0%  (baseline)")
    print(f"  This (W4A4 LLM QAD)      = {pc:.1f}%")

    per_task = eval_info.get("per_task", [])
    if per_task:
        print("\nPer-task:")
        for t in per_task:
            succ  = t["metrics"]["successes"]
            pct   = sum(succ) / len(succ) * 100
            fails = [i for i, s in enumerate(succ) if not s]
            print(f"  task {t['task_id']:2d}: {pct:5.1f}%  fails={fails}")

    scale_label = f"int{args.scale_bits}" if args.scale_bits else "fp16"
    result = {
        "stage": "stage8g_llm_qad",
        "group_size": args.group_size,
        "scale_label": scale_label,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "description": "Phase 1: W4A4 LLM QAD (DiT FP16 frozen)",
        "overall": overall,
        "eval_info": eval_info,
    }
    out_path = out_dir / f"libero_10_llm_qad_g{args.group_size}_{scale_label}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
    print(f"[INFO] Phase 2 command:")
    print(f"  python stage8f_qad.py --quant_format w4a4 --group_size {args.group_size} \\")
    print(f"      --llm_ckpt {best_ckpt} --teacher_cache <cache_path>")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
