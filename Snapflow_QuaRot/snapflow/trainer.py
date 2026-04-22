"""SnapFlow distillation trainer.

Usage:
    python scripts/stage1_snapflow_distill.py --config configs/stage1_snapflow.yaml

The trainer:
1. Loads pi0.5 base policy (teacher weights).
2. Creates a trainable student copy.
3. Trains student to match teacher's flow predictions in 1 NFE.
4. Saves student checkpoint to artifacts/stage1_student.safetensors.

TODOs before running:
- Implement snapflow/distill_loss.py::_get_velocity_at using PI05Pytorch.embed_prefix + .denoise_step
- Verify LIBERO dataset repo_id in configs/stage1_snapflow.yaml
- Tune train.steps, lr, batch_size for available GPU memory

Architecture access notes (modeling_pi05.py):
- PI05Pytorch.embed_prefix(images, img_masks, tokens, masks) → prefix embeddings
- PI05Pytorch.denoise_step(prefix_pad_masks, past_key_values, x_t, timestep) → v_t
- The KV cache (past_key_values) comes from:
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=..., position_ids=..., past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=True,
    )
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Path setup
_root = Path(__file__).resolve().parents[2]
for _p in [str(_root / "Snapflow_QuaRot"), str(_root / "lerobot" / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def train(cfg_path: str):
    from common.recipe import Recipe
    from common.policy_loader import load_policy
    from snapflow.teacher_student import TeacherStudent
    from snapflow.data import make_libero_dataloader

    recipe = Recipe.from_yaml(cfg_path)
    device = recipe.eval.device

    log.info("Loading base policy (teacher init)...")
    # Load WITHOUT applying snapflow weights (student starts from base)
    orig_enabled = recipe.snapflow.enabled
    recipe.snapflow.enabled = False
    policy, pre, post, env_pre, env_post = load_policy(recipe, device=device)
    recipe.snapflow.enabled = orig_enabled

    model = TeacherStudent(policy, share_weights=False).to(device)

    # Optimizer over student parameters only
    student_params = [p for p in model.student.parameters() if p.requires_grad]
    optimizer = AdamW(
        student_params,
        lr=1e-5,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    snapflow_cfg = recipe.snapflow
    train_steps = 10_000
    scheduler = CosineAnnealingLR(optimizer, T_max=train_steps, eta_min=1e-6)

    # Dataset
    dataset_repo_id = "lerobot/libero_10_lerobot"
    loader = make_libero_dataloader(
        dataset_repo_id=dataset_repo_id,
        batch_size=4,
        device=device,
    )
    loader_iter = iter(loader)

    log.info(f"Starting SnapFlow distillation: {train_steps} steps")

    for step in range(train_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

        loss_dict = model(batch)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_params, 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            log.info(
                f"Step {step}/{train_steps}  "
                f"loss={loss.item():.6f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    # Save student checkpoint
    out_path = Path(snapflow_cfg.student_ckpt or "artifacts/stage1_student.safetensors")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file
    state = {k: v for k, v in model.student.state_dict().items()}
    save_file(state, str(out_path))
    log.info(f"Student checkpoint saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage1_snapflow.yaml")
    args = parser.parse_args()
    train(args.config)
