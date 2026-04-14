"""
generate_summary_csv.py

모든 실험 결과를 하나의 summary CSV로 저장.
20 episodes 결과 우선, 5 episodes 결과도 포함 (n_episodes 컬럼으로 구분).
"""

import json, csv, math
from pathlib import Path
from collections import Counter

DIST_STATS_PATH = Path("logs/dist_analysis_v4/dist_stats.json")
SCALE_SHIFT = 24
FP32_BASELINE_SUCCESS = 90.0   # fp32 baseline (libero_spatial)


def bytes_quant(out_f, in_f, bits, block_size):
    pad = (-in_f) % block_size
    padded = in_f + pad
    n_blocks = padded // block_size
    actual = out_f * padded * 1 + out_f * n_blocks * 4        # int8 + int32 scale
    packed = math.ceil(out_f * padded * bits / 8) + out_f * n_blocks * 4
    return actual, packed


def summarize(result_path, weight_stats):
    p = Path(result_path)
    d = json.load(open(p))
    cfg = d.get("config", {})
    lc  = d.get("layer_configs", {})

    # ── 기본 정보 ────────────────────────────────────────────────────────
    n_ep      = cfg.get("n_episodes", "?")
    pc_succ   = d.get("pc_success", float("nan"))
    act_mode  = "wo" if not cfg.get("enable_act_quant") else cfg.get("act_quant_mode", "?")
    target    = cfg.get("target", "all")

    # ── bit-width 분포 ───────────────────────────────────────────────────
    lm_w  = Counter(v["w_bits"] for v in lc.values() if v["component"] == "lm")
    dit_w = Counter(v["w_bits"] for v in lc.values() if v["component"] == "dit")
    lm_a  = Counter(str(v["a_bits"]) for v in lc.values() if v["component"] == "lm")
    dit_a = Counter(str(v["a_bits"]) for v in lc.values() if v["component"] == "dit")

    n_lm  = sum(lm_w.values())
    n_dit = sum(dit_w.values())
    n_tot = n_lm + n_dit

    avg_w_lm  = sum(b*c for b,c in lm_w.items())  / max(n_lm, 1)
    avg_w_dit = sum(b*c for b,c in dit_w.items()) / max(n_dit, 1)
    avg_w_all = (avg_w_lm * n_lm + avg_w_dit * n_dit) / max(n_tot, 1)

    a_lm_list  = [v["a_bits"] for v in lc.values() if v["component"]=="lm"  and v["a_bits"] is not None]
    a_dit_list = [v["a_bits"] for v in lc.values() if v["component"]=="dit" and v["a_bits"] is not None]
    avg_a_lm   = sum(a_lm_list)  / len(a_lm_list)  if a_lm_list  else 0.0
    avg_a_dit  = sum(a_dit_list) / len(a_dit_list) if a_dit_list else 0.0
    avg_a_all  = sum(a_lm_list + a_dit_list) / len(a_lm_list + a_dit_list) if (a_lm_list or a_dit_list) else 0.0

    # block_size 분포
    blk_cnt = Counter(v["block_size"] for v in lc.values())
    dominant_blk = blk_cnt.most_common(1)[0][0] if blk_cnt else "-"

    # ── 메모리 계산 ──────────────────────────────────────────────────────
    total_fp32 = total_bf16 = total_actual = total_packed = 0
    total_base_flops = total_eff_w = total_eff_wa = 0
    total_params = 0

    for ln, lc_row in lc.items():
        ws = weight_stats.get(ln, {})
        sh = ws.get("shape")
        if not sh or len(sh) < 2:
            continue
        of, inf = int(sh[0]), int(sh[1])
        total_params += of * inf
        total_fp32   += of * inf * 4
        total_bf16   += of * inf * 2
        qa, qp = bytes_quant(of, inf, lc_row["w_bits"], lc_row["block_size"])
        total_actual += qa
        total_packed += qp

        bf = 2 * of * inf
        wr = lc_row["w_bits"] / 32.0
        ar = (lc_row["a_bits"] / 32.0) if lc_row["a_bits"] else 1.0
        total_base_flops += bf
        total_eff_w      += bf * wr
        total_eff_wa     += bf * wr * ar

    comp_fp32_actual = total_fp32   / total_actual if total_actual else 0
    comp_bf16_actual = total_bf16   / total_actual if total_actual else 0
    comp_fp32_packed = total_fp32   / total_packed if total_packed else 0
    comp_bf16_packed = total_bf16   / total_packed if total_packed else 0

    flops_w_pct  = total_eff_w  / total_base_flops * 100 if total_base_flops else 0
    flops_wa_pct = total_eff_wa / total_base_flops * 100 if total_base_flops else 0
    flops_w_reduction  = 100 - flops_w_pct
    flops_wa_reduction = 100 - flops_wa_pct

    acc_drop = round(pc_succ - FP32_BASELINE_SUCCESS, 1) if isinstance(pc_succ, (int, float)) else "-"

    return {
        # ── 식별 정보 ──
        "exp_name":        p.stem,
        "n_episodes":      n_ep,
        "task":            cfg.get("task", "?"),
        "target":          target,
        "act_mode":        act_mode,
        # ── 성능 ──
        "success_rate_%":          round(pc_succ, 1) if isinstance(pc_succ, float) else pc_succ,
        "acc_drop_vs_fp32_%":      acc_drop,
        # ── weight bit-width ──
        "avg_w_bits_all":          round(avg_w_all, 2),
        "avg_w_bits_lm":           round(avg_w_lm, 2),
        "avg_w_bits_dit":          round(avg_w_dit, 2),
        "lm_w_dist":               str(dict(sorted(lm_w.items()))),
        "dit_w_dist":              str(dict(sorted(dit_w.items()))),
        # ── activation bit-width ──
        "avg_a_bits_all":          round(avg_a_all, 2),
        "avg_a_bits_lm":           round(avg_a_lm, 2),
        "avg_a_bits_dit":          round(avg_a_dit, 2),
        "lm_a_dist":               str(dict(lm_a)),
        "dit_a_dist":              str(dict(dit_a)),
        # ── block size ──
        "dominant_block_size":     dominant_blk,
        "block_size_dist":         str(dict(sorted(blk_cnt.items()))),
        # ── 파라미터 수 ──
        "total_params_M":          round(total_params / 1e6, 2),
        "n_layers_quantized":      len(lc),
        # ── 메모리 (MB) ──
        "fp32_MB":                 round(total_fp32   / 1e6, 1),
        "bf16_MB":                 round(total_bf16   / 1e6, 1),
        "quant_actual_MB":         round(total_actual / 1e6, 1),
        "quant_packed_MB":         round(total_packed / 1e6, 1),
        "compression_vs_fp32_actual_x":  round(comp_fp32_actual, 2),
        "compression_vs_bf16_actual_x":  round(comp_bf16_actual, 2),
        "compression_vs_fp32_packed_x":  round(comp_fp32_packed, 2),
        "compression_vs_bf16_packed_x":  round(comp_bf16_packed, 2),
        # ── FLOPs (기준: FP32 = bit_ratio 1.0) ──
        "base_flops_GFLOPs":           round(total_base_flops / 1e9, 3),
        "eff_flops_w_only_GFLOPs":     round(total_eff_w  / 1e9, 3),
        "eff_flops_w_act_GFLOPs":      round(total_eff_wa / 1e9, 3),
        "flops_reduction_w_only_%":    round(flops_w_reduction,  1),
        "flops_reduction_w_act_%":     round(flops_wa_reduction, 1),
        "flops_ratio_w_only":          round(flops_w_pct  / 100, 4),
        "flops_ratio_w_act":           round(flops_wa_pct / 100, 4),
    }


def main():
    ds = json.load(open(DIST_STATS_PATH))
    weight_stats = ds.get("weight_stats", ds.get("weight", {}))

    all_jsons = sorted(Path("logs/mixed_int_quant").glob("*.json"))

    rows_20 = []
    rows_5  = []

    for p in all_jsons:
        try:
            row = summarize(p, weight_stats)
        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")
            continue
        if row["n_episodes"] == 20:
            rows_20.append(row)
        else:
            rows_5.append(row)

    # 성공률 내림차순 → avgW 오름차순 정렬
    def sort_key(r):
        s = r["success_rate_%"]
        w = r["avg_w_bits_all"]
        return (-s if isinstance(s, (int, float)) else 0, w)

    rows_20.sort(key=sort_key)
    rows_5.sort(key=sort_key)

    out_dir = Path("logs/mixed_int_quant")

    for rows, tag in [(rows_20, "20ep"), (rows_5, "5ep"), (rows_20 + rows_5, "all")]:
        if not rows:
            continue
        out_csv = out_dir / f"summary_{tag}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"저장: {out_csv}  ({len(rows)} rows)")

    # 터미널 요약 출력
    print(f"\n{'='*110}")
    print("  20-Episode 실험 결과 요약")
    print(f"  {'실험':<46} {'n':>3}  {'succ%':>6}  {'drop':>5}  {'avgW':>5}  {'avgA':>5}  "
          f"{'packed_MB':>10}  {'comp_bf16_pkg':>14}  {'flops_wa%':>10}  act_mode")
    print(f"  {'-'*115}")
    for r in rows_20:
        print(f"  {r['exp_name']:<46} {r['n_episodes']:>3}  "
              f"{str(r['success_rate_%'])+'%':>6}  "
              f"{str(r['acc_drop_vs_fp32_%']):>5}  "
              f"{r['avg_w_bits_all']:>5}  "
              f"{r['avg_a_bits_all']:>5}  "
              f"{r['quant_packed_MB']:>10.1f}  "
              f"{r['compression_vs_bf16_packed_x']:>14.2f}x  "
              f"{100-r['flops_reduction_w_act_%']:>9.1f}%  "
              f"{r['act_mode']}")


if __name__ == "__main__":
    main()
