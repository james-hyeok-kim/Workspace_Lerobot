"""Stage 10 — W4A16 LLM + W4A4 DiT QAD.

전략:
  LLM (PaliGemma + vision_tower + projector): W4A16
    - weight: INT4 blockwise (group_size=llm_group_size, 기본 4)
    - activation: FP16 (no quantization)
  DiT (gemma_expert): W4A4 g=8
    - weight+activation: INT4
  QAD: DiT only 학습 (LLM frozen), velocity MSE, stage8f 동일

프로토콜:
  [Step 1] FP16 policy 로드
  [Step 2] combined mtq.quantize (W4A16 LLM + W4A4 DiT, max calib)
           → LLM/DiT amax 모두 W4A16 LLM 분포 기준으로 calibrated
  [Step 3] stage8f best_student.pt 로드 (strict=False)
           → DiT weights: QAD 훈련 완료된 값으로 warm-start
           → DiT _amax: stage8f 값 덮어쓰기 (W4A16 ≈ FP16이므로 허용 범위)
           → LLM: stage8f에 LLM key 없음 → 변동 없음
  [Step 4] QAD training (DiT만 학습, LLM frozen)
  [Step 5] eval LIBERO-10

Usage:
  # Full run (500 steps, 10 tasks × 10 ep)
  python scripts/stage10_w4a16llm_w4a4dit_qad.py \\
    --init_ckpt results/stage8f_w4a4_g8/best_student.pt \\
    --output_dir results/stage10_w4a16llm_w4a4dit \\
    --n_steps 500

  # Smoke test (20 steps, task 0-1 × 2 ep)
  python scripts/stage10_w4a16llm_w4a4dit_qad.py \\
    --init_ckpt results/stage8f_w4a4_g8/best_student.pt \\
    --output_dir results/stage10_smoke \\
    --n_steps 20 --smoke
"""
import argparse
import json
import math
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


# ── quantization config ───────────────────────────────────────────────────────

def _build_combined_config(llm_group_size: int = 4, dit_group_size: int = 8, llm_weight_bits: int = 4, dit_act_bits: int = 4) -> dict:
    """WNA16 LLM + W4A{dit_act_bits} DiT combined config.

    주의: SequentialQuantizer 버그(v5 vision_tower 299 invalid) 방지 위해
    모든 entry는 단일 dict (list 사용 금지).
    """
    # LLM: weight INT{llm_weight_bits} blockwise, activation disabled
    w4_llm  = {"num_bits": llm_weight_bits, "block_sizes": {-1: llm_group_size}, "enable": True}
    # DiT: W4 group-wise weight, A{dit_act_bits} per-tensor activation
    w4_dit  = {"num_bits": 4, "block_sizes": {-1: dit_group_size}, "enable": True}
    act_dit = {"num_bits": dit_act_bits, "enable": True}
    fp16    = {"enable": False}

    return {
        "quant_cfg": {
            # LLM WNA16: weight only
            "*language_model*weight_quantizer":          w4_llm,
            "*vision_tower*weight_quantizer":            w4_llm,
            "*multi_modal_projector*weight_quantizer":   w4_llm,
            "*language_model*input_quantizer":           fp16,
            "*vision_tower*input_quantizer":             fp16,
            "*multi_modal_projector*input_quantizer":    fp16,
            # DiT W4A{dit_act_bits}
            "*gemma_expert*weight_quantizer":            w4_dit,
            "*gemma_expert*input_quantizer":             act_dit,
            # 항상 FP16
            "*lm_head*":                                 fp16,
            "*[kv]_bmm_quantizer":                       fp16,
            "default":                                   fp16,
        },
        "algorithm": "max",
    }


def _build_llm_only_config(llm_group_size: int = 4, weight_bits: int = 4) -> dict:
    """WNA16 LLM only — DiT stays FP16 (no gemma_expert entries)."""
    w_llm = {"num_bits": weight_bits, "block_sizes": {-1: llm_group_size}, "enable": True}
    fp16   = {"enable": False}
    return {
        "quant_cfg": {
            "*language_model*weight_quantizer":          w_llm,
            "*vision_tower*weight_quantizer":            w_llm,
            "*multi_modal_projector*weight_quantizer":   w_llm,
            "*language_model*input_quantizer":           fp16,
            "*vision_tower*input_quantizer":             fp16,
            "*multi_modal_projector*input_quantizer":    fp16,
            "*lm_head*":                                 fp16,
            "*[kv]_bmm_quantizer":                       fp16,
            "default":                                   fp16,
        },
        "algorithm": "max",
    }


def _build_dit_only_config(dit_group_size: int = 8, dit_act_bits: int = 4) -> dict:
    """W4A{dit_act_bits} DiT only — LLM stays FP16."""
    w4_dit  = {"num_bits": 4, "block_sizes": {-1: dit_group_size}, "enable": True}
    act_dit = {"num_bits": dit_act_bits, "enable": True}
    fp16    = {"enable": False}
    return {
        "quant_cfg": {
            "*gemma_expert*weight_quantizer": w4_dit,
            "*gemma_expert*input_quantizer":  act_dit,
            "*lm_head*":                      fp16,
            "*[kv]_bmm_quantizer":            fp16,
            "default":                        fp16,
        },
        "algorithm": "max",
    }


# ── calibration ───────────────────────────────────────────────────────────────

def _make_calib_loop(device: str, n_batches: int = 32):
    def calib_loop(model):
        model.eval()
        dataset = LiberoHDF5Dataset(CALIB_PATH, NORMALIZER_PATH, seed=0)
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
                except Exception as e:
                    print(f"[WARN] calib batch skipped: {e}")
                n += 1
                if n >= n_batches:
                    break
        print(f"[INFO] Calibration done: {n} batches")
    return calib_loop


def _verify_amax(policy, out_dir: Path):
    bad, ok = [], []
    llm_count = 0
    for name, mod in policy.named_modules():
        for qname in ("input_quantizer", "weight_quantizer"):
            q = getattr(mod, qname, None)
            if q is None:
                continue
            if not getattr(q, "is_enabled", False):
                continue
            if any(p in name for p in ("language_model", "vision_tower", "multi_modal_projector")):
                if qname == "weight_quantizer":
                    llm_count += 1
            amax = getattr(q, "_amax", None)
            if amax is None or (isinstance(amax, torch.Tensor) and (amax <= 0).any()):
                bad.append(f"{name}.{qname}")
            else:
                ok.append(f"{name}.{qname}")

    print(f"[INFO] Quantizer amax: {len(ok)} valid, {len(bad)} invalid")
    print(f"[INFO] LLM weight quantizers enabled: {llm_count} (기대: ~120-130)")
    if bad:
        print(f"[WARN] Invalid: {bad[:5]}{'...' if len(bad)>5 else ''}")
        for name, mod in policy.named_modules():
            for qname in ("input_quantizer", "weight_quantizer"):
                q = getattr(mod, qname, None)
                if q is None:
                    continue
                if not getattr(q, "is_enabled", False):
                    continue
                amax = getattr(q, "_amax", None)
                if amax is None or (isinstance(amax, torch.Tensor) and (amax <= 0).any()):
                    # ones_like preserves block-wise shape (e.g., [393216,1] not [1])
                    if isinstance(amax, torch.Tensor):
                        q._amax = torch.ones_like(amax)
                    else:
                        q._amax = torch.ones(1, device=next(policy.parameters()).device)
        print("[INFO] Patched invalid amax → 1.0 (shape preserved)")

    summary = {"valid": len(ok), "invalid": len(bad), "llm_weight_count": llm_count,
               "invalid_names": bad[:20]}
    with open(out_dir / "quantizer_amax_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return len(ok), len(bad)


# ── model utils ───────────────────────────────────────────────────────────────

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


def _freeze_for_qad(policy):
    freeze_patterns = [
        "language_model", "vision_tower", "multi_modal_projector",
        "layernorm.dense",  # AdaLN — freeze 안 하면 성능 붕괴 (stage8b 실패 원인)
        "lm_head", "action_in_proj", "action_out_proj",
    ]
    n_frozen = n_trainable = 0
    for name, param in policy.named_parameters():
        if any(p in name for p in freeze_patterns):
            param.requires_grad_(False)
            n_frozen += param.numel()
        else:
            param.requires_grad_(True)
            n_trainable += param.numel()
    print(f"[INFO] Trainable: {n_trainable/1e6:.1f}M  Frozen: {n_frozen/1e6:.1f}M")


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


# ── QAD training ──────────────────────────────────────────────────────────────

def _qad_train(student, teacher, device: str, n_steps: int, lr: float,
               batch_size: int, out_dir: Path, action_dim: int = 7):
    dataset = LiberoHDF5Dataset(CALIB_PATH, NORMALIZER_PATH, seed=42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, drop_last=True, pin_memory=True)
    data_iter = iter(loader)

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    student.train()
    teacher.eval()

    log_rows = []
    best_loss = float("inf")
    best_ckpt = out_dir / "best_student.pt"

    for step in range(1, n_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

        images, img_masks = student._preprocess_images(batch)
        tokens  = batch[OBS_LANGUAGE_TOKENS]
        masks   = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = student.prepare_action(batch)
        B = actions.shape[0]

        noise = student.model.sample_noise(actions.shape, device)
        t1    = torch.ones(B, device=device, dtype=actions.dtype)
        t05   = torch.full((B,), 0.5, device=device, dtype=actions.dtype)

        with torch.no_grad():
            v1  = get_velocity(teacher.model, images, img_masks, tokens, masks,
                               noise, t1, s=t1)
            x05 = noise - 0.5 * v1
            v05 = get_velocity(teacher.model, images, img_masks, tokens, masks,
                               x05, t05, s=t05)
            v_target = 0.5 * (v1 + v05)

        s0     = torch.zeros(B, device=device)
        v_pred = get_velocity(student.model, images, img_masks, tokens, masks,
                              noise, t1, s=s0)

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

    torch.save(student.state_dict(), out_dir / "final_student.pt")
    print(f"[INFO] QAD done. Best loss={best_loss:.6f}  saved → {best_ckpt}")

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(log_rows, f, indent=2)

    return best_ckpt, best_loss


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_ckpt", default="results/stage8f_w4a4_g8/best_student.pt",
                        help="DiT warm-start checkpoint (stage8f best_student.pt 권장)")
    parser.add_argument("--llm_group_size", type=int, default=4,
                        help="LLM W4A16 weight group size (ablation: g=4 → 78%)")
    parser.add_argument("--dit_group_size", type=int, default=8,
                        help="DiT W4A4 weight group size (stage8f: g=8 → 93%)")
    parser.add_argument("--calib_batches", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--n_action_steps", type=int, default=10,
                        help="Action steps per policy call (model default: 50)")
    parser.add_argument("--output_dir", default="results/stage10_w4a16llm_w4a4dit")
    parser.add_argument("--smoke", action="store_true",
                        help="smoke test: 20 steps, task 0-1 × 2 ep, amax 검증만")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_train_ckpt", default=None,
                        help="Override best_student.pt path for --skip_train (default: out_dir/best_student.pt)")
    parser.add_argument("--suite", default="libero_10",
                        help="LIBERO suite: libero_10 / libero_spatial / libero_object / libero_goal")
    parser.add_argument("--llm_only", action="store_true",
                        help="WNA16 PTQ on LLM only — DiT stays FP16, no QAD")
    parser.add_argument("--llm_weight_bits", type=int, default=4,
                        help="LLM weight quantization bits (default: 4, use 8 for W8A16)")
    parser.add_argument("--dit_act_bits", type=int, default=4,
                        help="DiT activation quantization bits (default: 4, use 8 for W4A8)")
    parser.add_argument("--fp16_only", action="store_true",
                        help="FP16 baseline mode — skip quantization, eval Path A directly")
    parser.add_argument("--dit_only", action="store_true",
                        help="FP16 LLM + W4A{dit_act_bits} DiT QAD — load checkpoint via --skip_train_ckpt")
    parser.add_argument("--n_envs", type=int, default=None,
                        help="Override n_envs (default = eval_episodes)")
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    if args.smoke:
        args.n_steps = 20
        args.eval_episodes = 2
        task_ids = [0, 1]
        print("[SMOKE] 20 steps, task 0-1 × 2 ep")
    else:
        task_ids = list(range(10))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import modelopt.torch.quantization as mtq

    # ── [Step 1] FP16 policy 로드 ─────────────────────────────────────────
    print("[INFO] Building policy from pretrained ...")
    policy_cfg = PreTrainedConfig.from_pretrained(PRETRAINED_PATH)
    policy_cfg.pretrained_path       = PRETRAINED_PATH
    policy_cfg.device                = args.device
    policy_cfg.use_amp               = False
    policy_cfg.compile_model         = False
    policy_cfg.gradient_checkpointing = False
    policy_cfg.n_action_steps        = args.n_action_steps
    policy_cfg.num_inference_steps   = args.num_inference_steps

    n_envs = args.n_envs if args.n_envs is not None else args.eval_episodes
    env_cfg = LiberoEnv(task=args.suite, task_ids=task_ids)
    envs_dict = make_env(env_cfg, n_envs=n_envs)

    student = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    _patch_embed_image(student)

    sd = torch.load(PATH_A, map_location="cpu")
    student.load_state_dict(sd, strict=False)
    print("[INFO] Path A (FP16) loaded into student")

    if args.fp16_only:
        # FP16 baseline: no quantization, just eval Path A
        best_loss = float("nan")
        print("[INFO] fp16_only mode — skipping quantization, evaluating FP16 directly")

    elif args.llm_only:
        # LLM-only WNA16 PTQ: quantize LLM, DiT stays FP16, no QAD
        cfg = _build_llm_only_config(args.llm_group_size, args.llm_weight_bits)
        calib_loop = _make_calib_loop(args.device, args.calib_batches)
        mtq.quantize(student, cfg, forward_loop=calib_loop)
        _verify_amax(student, out_dir)
        best_loss = float("nan")
        print(f"[INFO] llm_only mode — W{args.llm_weight_bits}A16 LLM (g={args.llm_group_size}), DiT FP16")

    elif args.dit_only:
        # DiT-only W4A{dit_act_bits} QAD: LLM stays FP16, load pre-trained DiT checkpoint
        cfg = _build_dit_only_config(args.dit_group_size, args.dit_act_bits)
        calib_loop = _make_calib_loop(args.device, args.calib_batches)
        mtq.quantize(student, cfg, forward_loop=calib_loop)
        _verify_amax(student, out_dir)
        best_ckpt_path = Path(args.skip_train_ckpt) if args.skip_train_ckpt else Path("results/stage8f_w4a4_g8/best_student.pt")
        best_sd = torch.load(best_ckpt_path, map_location="cpu")
        student.load_state_dict(best_sd, strict=False)
        best_loss = float("nan")
        print(f"[INFO] dit_only mode — FP16 LLM + W4A{args.dit_act_bits} DiT (g={args.dit_group_size}), loaded {best_ckpt_path}")

    elif not args.skip_train:
        # ── [Step 2] Combined quantization + max calib ────────────────────
        print(f"[Step 2] Combined quant: W{args.llm_weight_bits}A16 LLM (g={args.llm_group_size}) + "
              f"W4A{args.dit_act_bits} DiT (g={args.dit_group_size}), max calib ...")
        cfg = _build_combined_config(args.llm_group_size, args.dit_group_size, args.llm_weight_bits, args.dit_act_bits)
        calib_loop = _make_calib_loop(args.device, args.calib_batches)
        mtq.quantize(student, cfg, forward_loop=calib_loop)
        print(f"[Step 2] Done.")

        # amax 검증
        n_valid, n_invalid = _verify_amax(student, out_dir)

        # ── [Step 3] init_ckpt warm-start (DiT weights 복원) ──────────────
        if args.init_ckpt:
            print(f"[Step 3] Loading init_ckpt for DiT warm-start: {args.init_ckpt}")
            ckpt = torch.load(args.init_ckpt, map_location="cpu")
            state = ckpt.get("model", ckpt)
            # Filter shape-mismatch keys (load_state_dict strict=False still raises on shape mismatch)
            model_state = student.state_dict()
            filtered, skipped = {}, []
            for k, v in state.items():
                if k in model_state and v.shape != model_state[k].shape:
                    skipped.append(f"{k}: ckpt={tuple(v.shape)} vs model={tuple(model_state[k].shape)}")
                else:
                    filtered[k] = v
            if skipped:
                print(f"[Step 3] Skipped {len(skipped)} shape-mismatch key(s): {skipped[:5]}")
            missing, unexpected = student.load_state_dict(filtered, strict=False)
            # 기대: unexpected=0 (quantizer key 매칭), missing=LLM quantizer key들
            dit_restored = sum(1 for k in filtered if "gemma_expert" in k and "_amax" not in k)
            dit_amax_restored = sum(1 for k in filtered if "gemma_expert" in k and "_amax" in k)
            print(f"[Step 3] missing={len(missing)}, unexpected={len(unexpected)}")
            print(f"[Step 3] DiT weights restored: ~{dit_restored}, "
                  f"DiT amax restored: {dit_amax_restored}")

        if args.smoke:
            print("[SMOKE] Quantization + warm-start OK. Skip training.")
            return

        # ── FP16 teacher 로드 ─────────────────────────────────────────────
        print("[INFO] Building FP16 teacher ...")
        teacher = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
        _patch_embed_image(teacher)
        teacher.load_state_dict(sd, strict=False)
        teacher.to(args.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # ── [Step 4] QAD training ─────────────────────────────────────────
        _freeze_for_qad(student)
        _reset_dynamo(student)
        student.to(args.device)

        print(f"\n[INFO] QAD: W{args.llm_weight_bits}A16 LLM (g={args.llm_group_size}) + "
              f"W4A{args.dit_act_bits} DiT (g={args.dit_group_size}) | steps={args.n_steps} | lr={args.lr}")
        best_ckpt, best_loss = _qad_train(
            student=student,
            teacher=teacher,
            device=args.device,
            n_steps=args.n_steps,
            lr=args.lr,
            batch_size=args.train_batch_size,
            out_dir=out_dir,
        )

        del teacher
        torch.cuda.empty_cache()

    else:
        # skip_train: 저장된 checkpoint 로드
        best_ckpt = Path(args.skip_train_ckpt) if args.skip_train_ckpt else out_dir / "best_student.pt"
        best_loss = float("nan")
        print(f"[INFO] skip_train: applying quant and loading {best_ckpt}")
        cfg = _build_combined_config(args.llm_group_size, args.dit_group_size, args.llm_weight_bits, args.dit_act_bits)
        calib_loop = _make_calib_loop(args.device, args.calib_batches)
        mtq.quantize(student, cfg, forward_loop=calib_loop)
        _verify_amax(student, out_dir)
        best_sd = torch.load(best_ckpt, map_location="cpu")
        student.load_state_dict(best_sd, strict=False)

    # ── [Step 5] eval ─────────────────────────────────────────────────────
    student.eval()
    _reset_dynamo(student)
    student.to(args.device)

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

    print(f"\n[INFO] Evaluating {len(task_ids)} tasks × {args.eval_episodes} ep "
          f"(NFE={args.num_inference_steps}) ...")
    with torch.no_grad():
        eval_info = eval_policy_all(
            envs=envs_dict,
            policy=student,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.eval_episodes,
            start_seed=1000,
        )

    for suite_envs in envs_dict.values():
        for env in suite_envs.values():
            try: env.close()
            except Exception: pass

    overall = eval_info.get("overall", {})
    pc = overall.get("pc_success", float("nan"))

    if args.fp16_only:
        mode_tag = "fp16"
    elif args.llm_only:
        mode_tag = f"llm_only_w{args.llm_weight_bits}g{args.llm_group_size}"
    elif args.dit_only:
        mode_tag = f"fp16llm_w4a{args.dit_act_bits}dit_g{args.dit_group_size}_qad"
    else:
        mode_tag = f"w{args.llm_weight_bits}a16llm_g{args.llm_group_size}_w4a{args.dit_act_bits}dit_qad"
    print(f"\n{'='*60}")
    if args.fp16_only:
        print(f"[FP16 Baseline] suite={args.suite}")
    elif args.llm_only:
        print(f"[LLM-only PTQ] W{args.llm_weight_bits}A16 LLM (g={args.llm_group_size}), DiT FP16 | suite={args.suite}")
    elif args.dit_only:
        print(f"[DiT-only QAD] FP16 LLM + W4A{args.dit_act_bits} DiT (g={args.dit_group_size}) | suite={args.suite}")
    else:
        print(f"[Stage10] W{args.llm_weight_bits}A16 LLM (g={args.llm_group_size}) + W4A{args.dit_act_bits} DiT (g={args.dit_group_size}) | suite={args.suite}")
        if not (args.skip_train or math.isnan(best_loss)):
            print(f"  best_loss (QAD):  {best_loss:.6f}")
    print(f"  pc_success:       {pc:.1f}%")

    per_task = eval_info.get("per_task", [])
    if per_task:
        print("\nPer-task:")
        for t in per_task:
            succ = t["metrics"]["successes"]
            pct  = sum(succ) / len(succ) * 100
            fails = [i for i, s in enumerate(succ) if not s]
            print(f"  task {t['task_id']:2d}: {pct:5.1f}%  fails={fails}")

    result = {
        "stage": ("fp16_baseline" if args.fp16_only else
                  ("llm_only_ptq" if args.llm_only else
                  ("fp16llm_dit_only_w4a4_qad" if args.dit_only else
                   "stage10_w4a16llm_w4a4dit_qad"))),
        "suite": args.suite,
        "mode": mode_tag,
        "llm_quant": "FP16" if (args.fp16_only or args.dit_only) else f"W{args.llm_weight_bits}A16 INT{args.llm_weight_bits} blockwise-{args.llm_group_size}",
        "dit_quant": "FP16" if (args.fp16_only or args.llm_only) else f"W4A{args.dit_act_bits} g={args.dit_group_size} QAD",
        "init_ckpt": None if (args.fp16_only or args.llm_only) else (str(args.skip_train_ckpt) if args.dit_only else args.init_ckpt),
        "n_steps": 0 if (args.fp16_only or args.llm_only or args.dit_only) else args.n_steps,
        "best_loss": best_loss,
        "num_inference_steps": args.num_inference_steps,
        "n_action_steps": args.n_action_steps,
        "overall": overall,
        "eval_info": eval_info,
    }
    out_path = out_dir / f"{args.suite}_{mode_tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
