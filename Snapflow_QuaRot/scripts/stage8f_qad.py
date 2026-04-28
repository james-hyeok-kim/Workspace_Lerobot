"""Stage 8f — QAD: Quantization-Aware Distillation.

SnapFlow 오차 + 양자화 오차를 단일 학습으로 동시 최소화.

Teacher : Path A (FP16, frozen) — NFE=10 기준 velocity (2-step Euler)
Student : Path A + fake-quant (INT4 g=8 또는 NVFP4 b=8) — NFE=1 예측
Loss    : MSE(v_student_NFE1, v_target_NFE10)
Trainable: gemma_expert (attn+mlp), AdaLN(layernorm.dense)는 완전 freeze

기존 stage8b 실패 원인: AdaLN drift (lr=5e-4, 500 steps)
QAD fix: layernorm.dense 완전 freeze + lr=1e-4

Usage:
  python stage8f_qad.py --quant_format int4   --output_dir results/stage8f_int4_g8
  python stage8f_qad.py --quant_format nvfp4  --output_dir results/stage8f_nvfp4_b8
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

def _build_int4_config(group_size: int = 8, scale_bits: int = None) -> dict:
    block_sizes = {-1: group_size}
    if scale_bits is not None:
        block_sizes["scale_bits"] = scale_bits  # INT16 scale → INT×INT arithmetic
    w4 = {"num_bits": 4, "block_sizes": block_sizes, "enable": True}
    fp16 = {"enable": False}
    return {
        "quant_cfg": {
            "*weight_quantizer": w4,
            "*input_quantizer": fp16,
            "*lm_head*weight_quantizer": fp16,
            "*action_in_proj*weight_quantizer": fp16,
            "*action_out_proj*weight_quantizer": fp16,
            "*[kv]_bmm_quantizer": fp16,
            "default": fp16,
        },
        "algorithm": "max",
    }


def _build_nvfp4_config(block_size: int = 8) -> dict:
    # NVFP4 E2M1: type="dynamic" + scale_bits=(4,3) → dynamic_block_quant 경로
    # type="static"은 scaled_e4m3()로 가서 E2M1 미지원 에러 발생하므로 주의
    fp4 = {
        "num_bits": (2, 1),
        "block_sizes": {-1: block_size, "type": "dynamic", "scale_bits": (4, 3)},
        "enable": True,
    }
    fp16 = {"enable": False}
    return {
        "quant_cfg": {
            "*weight_quantizer": fp4,
            "*input_quantizer": fp16,
            "*lm_head*weight_quantizer": fp16,
            "*action_in_proj*weight_quantizer": fp16,
            "*action_out_proj*weight_quantizer": fp16,
            "*[kv]_bmm_quantizer": fp16,
            "default": fp16,
        },
        "algorithm": "max",
    }


def _build_w4a4_config(group_size: int = 8, scale_bits: int = None,
                       quant_scope: str = "expert") -> dict:
    block_sizes = {-1: group_size}
    if scale_bits is not None:
        block_sizes["scale_bits"] = scale_bits
    w4   = {"num_bits": 4, "block_sizes": block_sizes, "enable": True}
    a4   = {"num_bits": 4, "enable": True}
    fp16 = {"enable": False}

    if quant_scope == "full":
        # LLM (PaliGemma) + DiT (GemmaExpert) 모두 W4A4
        # QAD 학습은 DiT만 (LLM은 freeze 유지)
        return {
            "quant_cfg": {
                "*language_model*weight_quantizer": w4,
                "*vision_tower*weight_quantizer": w4,
                "*multi_modal_projector*weight_quantizer": w4,
                "*gemma_expert*weight_quantizer": w4,
                "*language_model*input_quantizer": a4,
                "*vision_tower*input_quantizer": a4,
                "*multi_modal_projector*input_quantizer": a4,
                "*gemma_expert*input_quantizer": a4,
                "*lm_head*": fp16,
                "*[kv]_bmm_quantizer": fp16,
                "default": fp16,
            },
            "algorithm": "max",
        }
    else:
        # expert only (기존 동작)
        return {
            "quant_cfg": {
                "*gemma_expert*weight_quantizer": w4,
                "*gemma_expert*input_quantizer": a4,
                "*[kv]_bmm_quantizer": fp16,
                "default": fp16,
            },
            "algorithm": "max",
        }


def _build_ablation_w4a4_config(ablation_mode: str, group_size: int) -> dict:
    """Component-specific W4A4 config for ablation QAD experiments."""
    w4 = {"num_bits": 4, "block_sizes": {-1: group_size}, "enable": True}
    a4 = {"num_bits": 4, "enable": True}
    fp16 = {"enable": False}

    if ablation_mode == "expert_attn":
        return {
            "quant_cfg": {
                "*gemma_expert*q_proj*weight_quantizer": w4,
                "*gemma_expert*k_proj*weight_quantizer": w4,
                "*gemma_expert*v_proj*weight_quantizer": w4,
                "*gemma_expert*o_proj*weight_quantizer": w4,
                "*gemma_expert*q_proj*input_quantizer": a4,
                "*gemma_expert*k_proj*input_quantizer": a4,
                "*gemma_expert*v_proj*input_quantizer": a4,
                "*gemma_expert*o_proj*input_quantizer": a4,
                "*[kv]_bmm_quantizer": fp16,
                "default": fp16,
            },
            "algorithm": "max",
        }
    elif ablation_mode == "expert_mlp":
        return {
            "quant_cfg": {
                "*gemma_expert*gate_proj*weight_quantizer": w4,
                "*gemma_expert*up_proj*weight_quantizer": w4,
                "*gemma_expert*down_proj*weight_quantizer": w4,
                "*gemma_expert*gate_proj*input_quantizer": a4,
                "*gemma_expert*up_proj*input_quantizer": a4,
                "*gemma_expert*down_proj*input_quantizer": a4,
                "*[kv]_bmm_quantizer": fp16,
                "default": fp16,
            },
            "algorithm": "max",
        }
    else:
        raise ValueError(f"Unknown ablation_mode: {ablation_mode}")


def _apply_fake_quant(policy, quant_format: str, group_size: int, device: str,
                      scale_bits: int = None, ablation_mode: str = None,
                      quant_scope: str = "expert"):
    import modelopt.torch.quantization as mtq

    if quant_format == "int4":
        cfg = _build_int4_config(group_size, scale_bits=scale_bits)
        scale_label = f"INT{scale_bits}" if scale_bits else "FP16"
    elif quant_format == "w4a4":
        if ablation_mode:
            cfg = _build_ablation_w4a4_config(ablation_mode, group_size)
            scale_label = f"W4A4-{ablation_mode}"
        else:
            cfg = _build_w4a4_config(group_size, scale_bits=scale_bits, quant_scope=quant_scope)
            scope_tag = "-full" if quant_scope == "full" else ""
            scale_label = f"W4A4{scope_tag}-INT{scale_bits}" if scale_bits else f"W4A4{scope_tag}"
    else:
        cfg = _build_nvfp4_config(group_size)
        scale_label = "FP4"

    print(f"[INFO] Applying fake-quant: {quant_format} g/b={group_size}, scope={quant_scope}, scale={scale_label}")

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
                except Exception:
                    pass
                n += 1
                if n >= 32:
                    break
        print(f"[INFO] Calibration done: {n} batches")

    mtq.quantize(policy, cfg, forward_loop=calib_loop)

    # Verify all enabled quantizers have valid amax (> 0)
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
        print(f"[WARN] Invalid amax in: {bad[:5]}{'...' if len(bad)>5 else ''}")
        # Force valid amax=1.0 for uncalibrated quantizers to avoid CUDA assert
        for name, mod in policy.named_modules():
            for qname in ("input_quantizer", "weight_quantizer"):
                q = getattr(mod, qname, None)
                if q is None:
                    continue
                amax = getattr(q, "_amax", None)
                if amax is None or (isinstance(amax, torch.Tensor) and (amax < 0).any()):
                    q._amax = torch.ones(1, device=next(policy.parameters()).device)
        print("[INFO] Patched invalid amax → 1.0")

    print(f"[INFO] fake-quant registered: {quant_format}")


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


def _freeze_for_qad(policy):
    """LLM + AdaLN freeze, gemma_expert (attn+mlp) 학습."""
    freeze_patterns = [
        "language_model",
        "vision_tower",
        "multi_modal_projector",
        "layernorm.dense",   # AdaLN timestep conditioning — stage8b 실패 원인
        "lm_head",
        "action_in_proj",
        "action_out_proj",
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


# ── QAD training loop ────────────────────────────────────────────────────────

def _load_teacher_cache(cache_path: str, device: str):
    """Load precomputed teacher velocity targets from disk."""
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


def _qad_train(student, teacher, device: str, n_steps: int, lr: float,
               batch_size: int, out_dir: Path, action_dim: int = 7,
               teacher_cache: str = None):
    """
    Teacher (FP16, frozen): 2-step Euler → v_target (NFE=10 근사)
    Student (fake-quant):   NFE=1 → v_pred
    Loss: MSE(v_pred, v_target)

    teacher_cache: path to precomputed cache from precompute_teacher_cache.py.
      If set, teacher forward is skipped entirely — cache batches are used directly.
      Teacher model can be None when cache is provided.
    """
    # ── Data source: cache or live dataset ───────────────────────────────────
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
    best_ckpt = out_dir / "best_student.pt"

    for step in range(1, n_steps + 1):
        if use_cache:
            # ── Cached path: load precomputed teacher targets ─────────────
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
            # ── Live path: run teacher forward each step ──────────────────
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

        # ── Student: fake-quant NFE=1 ─────────────────────────────────────
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
    print(f"[INFO] Best loss={best_loss:.6f}  saved → {best_ckpt}")

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(log_rows, f, indent=2)

    return best_ckpt


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_format", choices=["int4", "nvfp4", "w4a4"], required=True)
    parser.add_argument("--group_size", type=int, default=8,
                        help="group_size for INT4 / block_size for NVFP4")
    parser.add_argument("--scale_bits", type=int, default=None,
                        help="scale precision bits for INT4 (None=FP16, 16=INT16)")
    parser.add_argument("--quant_scope", choices=["expert", "full"], default="expert",
                        help="expert=DiT only W4A4; full=LLM+DiT both W4A4 (QAD trains DiT only)")
    parser.add_argument("--ablation_mode", choices=["expert_attn", "expert_mlp"], default=None,
                        help="w4a4 component ablation (only used with --quant_format w4a4)")
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=10)
    parser.add_argument("--task_ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_train", action="store_true",
                        help="skip training, load best_student.pt and eval only")
    parser.add_argument("--teacher_cache", default=None,
                        help="path to precomputed teacher cache (.pt) from precompute_teacher_cache.py. "
                             "If set, teacher model is not loaded — saves ~30 GB GPU memory.")
    parser.add_argument("--llm_ckpt", default=None,
                        help="Phase 2 모드: stage8g_llm_qad.py의 best_llm_student.pt 경로. "
                             "제공시 W4A4 LLM (Phase 1 학습된) + W4A4 DiT (Phase 2 학습) 구조. "
                             "LLM quantizer 상태를 Phase 1에서 복원하고 DiT만 QAD 학습.")
    args = parser.parse_args()

    scale_suffix    = f"_s{args.scale_bits}" if args.scale_bits else ""
    scope_suffix    = "_full" if args.quant_scope == "full" else ""
    ablation_suffix = f"_{args.ablation_mode}" if args.ablation_mode else ""
    phase2_suffix   = "_phase2" if args.llm_ckpt else ""
    out_dir = Path(args.output_dir or
                   f"results/stage8f_{args.quant_format}_g{args.group_size}"
                   f"{scope_suffix}{scale_suffix}{ablation_suffix}{phase2_suffix}")
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

    print("[INFO] Building student policy ...")
    env_cfg = LiberoEnv(task="libero_10", task_ids=args.task_ids)
    envs_dict = make_env(env_cfg, n_envs=args.eval_batch_size)

    student = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    _patch_embed_image(student)

    sd = torch.load(PATH_A, map_location="cpu")
    student.load_state_dict(sd, strict=False)
    print("[INFO] Path A loaded into student")

    if not args.skip_train:
        # ── build FP16 teacher (skip if cache provided) ───────────────────
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

        # ── apply fake-quant to student ───────────────────────────────────
        if args.llm_ckpt:
            # Phase 2: LLM+DiT 모두 W4A4, Phase 1 학습된 LLM 가중치 복원
            print(f"[INFO] Phase 2 mode: applying full W4A4 (LLM+DiT), "
                  f"then loading Phase 1 LLM from {args.llm_ckpt}")
            _apply_fake_quant(student, "w4a4", args.group_size, args.device,
                              scale_bits=args.scale_bits, quant_scope="full")
            llm_sd = torch.load(args.llm_ckpt, map_location="cpu")
            student.load_state_dict(llm_sd, strict=False)
            # strict=False: LLM 가중치+quantizer → Phase 1 값으로 복원
            #               DiT quantizer → 방금 calibration 값 유지 (FP16 Path A 기준)
            print(f"[INFO] Phase 1 LLM weights loaded. "
                  f"DiT quantizers calibrated from Path A.")
        else:
            _apply_fake_quant(student, args.quant_format, args.group_size, args.device,
                              scale_bits=args.scale_bits, ablation_mode=args.ablation_mode,
                              quant_scope=args.quant_scope)
        _freeze_for_qad(student)  # LLM freeze, DiT 학습 — Phase 1/2 모두 동일
        _reset_dynamo(student)
        student.to(args.device)

        print(f"\n[INFO] QAD: {args.quant_format} g={args.group_size} scope={args.quant_scope} | "
              f"steps={args.n_steps} | lr={args.lr}")
        best_ckpt = _qad_train(
            student=student,
            teacher=teacher,
            device=args.device,
            n_steps=args.n_steps,
            lr=args.lr,
            batch_size=args.train_batch_size,
            out_dir=out_dir,
            teacher_cache=args.teacher_cache,
        )

        # unload teacher to free memory
        if teacher is not None:
            del teacher
        torch.cuda.empty_cache()

    else:
        best_ckpt = out_dir / "best_student.pt"
        print(f"[INFO] skip_train: loading {best_ckpt}")
        _apply_fake_quant(student, args.quant_format, args.group_size, args.device,
                          scale_bits=args.scale_bits, ablation_mode=args.ablation_mode,
                          quant_scope=args.quant_scope)
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

    print(f"\n[INFO] Evaluating {len(args.task_ids)} tasks × {args.n_episodes} ep ...")
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
    pc = overall.get("pc_success", float("nan"))

    print(f"\n{'='*60}")
    print(f"[QAD {args.quant_format} g={args.group_size}] pc_success={pc:.1f}%")
    print(f"  Reference (PTQ only):")
    print(f"    FP16 (Path A):          100.0%")
    print(f"    INT4 g=8  PTQ:           71.0%")
    print(f"    NVFP4 b=8 PTQ:           19.0%")
    print(f"  QAD result ({args.quant_format} g={args.group_size}): {pc:.1f}%")

    per_task = eval_info.get("per_task", [])
    if per_task:
        print("\nPer-task:")
        for t in per_task:
            succ = t["metrics"]["successes"]
            pct  = sum(succ) / len(succ) * 100
            fails = [i for i, s in enumerate(succ) if not s]
            print(f"  task {t['task_id']:2d}: {pct:5.1f}%  fails={fails}")

    scale_label = f"int{args.scale_bits}" if args.scale_bits else "fp16"
    result = {
        "stage": "stage8f_qad",
        "quant_format": args.quant_format,
        "group_size": args.group_size,
        "quant_scope": args.quant_scope,
        "ablation_mode": args.ablation_mode,
        "scale_bits": args.scale_bits,
        "scale_label": scale_label,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "overall": overall,
        "eval_info": eval_info,
        "reference": {
            "fp16": 100.0,
            "int4_g8_ptq": 71.0,
            "nvfp4_b8_ptq": 19.0,
        },
    }
    out_path = out_dir / f"libero_10_qad_{args.quant_format}_g{args.group_size}{scope_suffix}_{scale_label}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
