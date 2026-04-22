"""Append-only JSONL result database + leaderboard.md generator."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

_DB_PATH = Path("results/results.jsonl")
_LEADERBOARD_PATH = Path("results/leaderboard.md")

_HEADER = (
    "| Stage | pc_success (%) | avg_sum_reward | latency_p50 (ms) | latency_p95 (ms) | NFE | n_action_steps | W_bits | A_bits | SnapFlow | QuaRot | OHB | W4A4 |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


class ResultsDB:
    def __init__(
        self,
        db_path: str | Path = _DB_PATH,
        leaderboard_path: str | Path = _LEADERBOARD_PATH,
    ):
        self.db_path = Path(db_path)
        self.lb_path = Path(leaderboard_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, stage_name: str, result: dict, recipe=None):
        """Append one result row and regenerate leaderboard."""
        agg = result.get("aggregated", {})
        lat = result.get("latency", {})
        cfg = result.get("config", {})

        row = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage_name,
            "pc_success": agg.get("pc_success"),
            "avg_sum_reward": agg.get("avg_sum_reward"),
            "avg_max_reward": agg.get("avg_max_reward"),
            "latency_p50_ms": lat.get("p50_ms"),
            "latency_p95_ms": lat.get("p95_ms"),
            "nfe": cfg.get("nfe"),
            "n_action_steps": cfg.get("n_action_steps"),
            "snapflow": cfg.get("snapflow", False),
            "quarot": cfg.get("quarot", False),
            "quarot_scope": cfg.get("quarot_scope", "none"),
            "ohb": cfg.get("ohb", False),
            "w4a4": cfg.get("w4a4", False),
            "w_bits": "4" if cfg.get("w4a4") else "16",
            "a_bits": "4" if cfg.get("w4a4") else "16",
        }

        with open(self.db_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        self._regenerate_leaderboard()
        log.info(f"ResultsDB: appended {stage_name}, pc_success={row['pc_success']}")

    def _load_all(self) -> list[dict]:
        if not self.db_path.exists():
            return []
        rows = []
        with open(self.db_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _regenerate_leaderboard(self):
        rows = self._load_all()

        # Deduplicate: keep last entry per stage name
        seen: dict[str, dict] = {}
        for r in rows:
            seen[r["stage"]] = r
        rows = list(seen.values())

        lines = ["# Leaderboard\n\n", _HEADER]
        for r in rows:
            pc = f"{r['pc_success']:.1f}" if r["pc_success"] is not None else "N/A"
            rew = f"{r['avg_sum_reward']:.4f}" if r["avg_sum_reward"] is not None else "N/A"
            lp50 = f"{r['latency_p50_ms']:.1f}" if r["latency_p50_ms"] is not None else "N/A"
            lp95 = f"{r['latency_p95_ms']:.1f}" if r["latency_p95_ms"] is not None else "N/A"
            lines.append(
                f"| {r['stage']} | {pc} | {rew} | {lp50} | {lp95} | {r['nfe']} | {r['n_action_steps']} "
                f"| {r['w_bits']} | {r['a_bits']} | {'✓' if r['snapflow'] else '—'} "
                f"| {'✓' if r['quarot'] else '—'} | {'✓' if r['ohb'] else '—'} "
                f"| {'✓' if r['w4a4'] else '—'} |\n"
            )

        with open(self.lb_path, "w") as f:
            f.writelines(lines)
