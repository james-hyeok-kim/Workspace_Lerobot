"""
generate_quant_report_csv.py

Mixed INT quantization 결과를 per-layer CSV로 정리.

컬럼:
  layer_name, component, layer_type, attn_mlp,
  out_features, in_features, param_count,
  w_bits, a_bits, block_size,
  fp32_bytes, bf16_bytes,
  quant_w_bytes (W_int + scale_i32), quant_total_bytes (+ bias),
  weight_compression_vs_fp32, weight_compression_vs_bf16,
  base_flops,
  w_flops_ratio  (w_bits / 32 — 선형 비율),
  a_flops_ratio  (a_bits / 32 if quantized, else 1.0),
  wa_flops_ratio (w × a, 둘 다 INT인 경우 곱),
  effective_flops_w_only,
  effective_flops_wa,

FLOPs 비율 정의 (사용자 요청):
  FP32 기준 = 32, INT8 = 8 → ratio = bits/32
  weight-only 레이어는 activation은 FP32(1.0) 그대로 유지
"""

import json
import csv
import math
import argparse
from pathlib import Path

DIST_STATS_PATH = Path("logs/dist_analysis_v4/dist_stats.json")
SCALE_SHIFT = 24  # INT32 scale: scale_fp × 2^24


def bytes_for_quant_weight(out_f: int, in_f: int, bits: int, block_size: int):
    """
    W_int + scale_i32 저장 바이트 수.

    Returns:
        actual_bytes : 현재 구현 (int8 storage, 1 byte/elem)
        packed_bytes : 이론 bit-packing (bits/8 bytes/elem)
    """
    pad = (-in_f) % block_size
    padded = in_f + pad
    num_blocks = padded // block_size
    scale_bytes = out_f * num_blocks * 4          # int32 = 4 bytes/elem

    # 현재 구현: W_int는 int8 텐서 (bit-width 무관하게 1 byte/elem)
    actual_bytes = out_f * padded * 1 + scale_bytes

    # 이론 bit-packing: bits/8 bytes/elem (e.g. INT4 → 0.5 byte/elem)
    import math
    packed_w_bytes = math.ceil(out_f * padded * bits / 8)
    packed_bytes   = packed_w_bytes + scale_bytes

    return actual_bytes, packed_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_jsons", nargs="+",
        default=[
            "logs/mixed_int_quant/mixed_int_all_lm4_dit4_per_block.json",
            "logs/mixed_int_quant/mixed_int_all_lm6_dit4_per_block.json",
            "logs/mixed_int_quant/mixed_int_all_lm1_dit1_per_block.json",
            "logs/mixed_int_quant/mixed_int_all_lm8_dit4_wo.json",
        ],
        help="분석할 result JSON 파일들"
    )
    parser.add_argument(
        "--dist_stats", default=str(DIST_STATS_PATH),
        help="dist_stats.json 경로"
    )
    parser.add_argument(
        "--output_dir", default="logs/mixed_int_quant",
        help="CSV 출력 디렉토리"
    )
    args = parser.parse_args()

    # dist_stats 로드 (shape 정보)
    ds = json.load(open(args.dist_stats))
    weight_stats = ds.get("weight_stats", ds.get("weight", {}))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for result_path in args.result_jsons:
        p = Path(result_path)
        if not p.exists():
            print(f"[SKIP] {p} not found")
            continue

        result = json.load(open(p))
        layer_configs = result.get("layer_configs", {})
        cfg = result.get("config", {})
        exp_name = p.stem
        pc_success = result.get("pc_success", "?")

        rows = []
        total_fp32 = total_bf16 = total_quant_actual = total_quant_packed = 0
        total_base_flops = total_eff_flops_w = total_eff_flops_wa = 0

        for layer_name, lc in layer_configs.items():
            wstat = weight_stats.get(layer_name, {})
            shape = wstat.get("shape", None)
            if shape is None or len(shape) < 2:
                continue

            out_f, in_f = int(shape[0]), int(shape[1])
            param_count = out_f * in_f

            w_bits    = int(lc["w_bits"])
            a_bits    = lc["a_bits"]           # int or None
            block_size = int(lc["block_size"])
            comp      = lc["component"]
            ltype     = lc["layer_type"]
            attn_mlp  = lc["attn_mlp"]

            # ── 메모리 계산 ─────────────────────────────────────────────────
            fp32_bytes  = param_count * 4
            bf16_bytes  = param_count * 2
            quant_actual_b, quant_packed_b = bytes_for_quant_weight(out_f, in_f, w_bits, block_size)

            comp_fp32_actual  = round(fp32_bytes / quant_actual_b, 3)
            comp_bf16_actual  = round(bf16_bytes / quant_actual_b, 3)
            comp_fp32_packed  = round(fp32_bytes / quant_packed_b, 3)
            comp_bf16_packed  = round(bf16_bytes / quant_packed_b, 3)

            # ── FLOPs 계산 ──────────────────────────────────────────────────
            # 기본 FLOPs = 2 × out_f × in_f (MAC = multiply + accumulate)
            base_flops = 2 * out_f * in_f

            w_ratio  = w_bits / 32.0            # weight precision ratio vs FP32
            a_ratio  = (a_bits / 32.0) if a_bits is not None else 1.0

            # weight-only INT: activation은 fp로 남음 → w_ratio만 적용
            eff_flops_w  = base_flops * w_ratio
            # weight+act 모두 INT: 둘 다 곱 (가능 HW에서)
            eff_flops_wa = base_flops * w_ratio * a_ratio

            wa_ratio = round(w_ratio * a_ratio, 6)

            # accumulate totals
            total_fp32         += fp32_bytes
            total_bf16         += bf16_bytes
            total_quant_actual += quant_actual_b
            total_quant_packed += quant_packed_b
            total_base_flops   += base_flops
            total_eff_flops_w  += eff_flops_w
            total_eff_flops_wa += eff_flops_wa

            rows.append({
                "layer_name":     layer_name,
                "component":      comp,
                "layer_type":     ltype,
                "attn_mlp":       attn_mlp,
                "out_features":   out_f,
                "in_features":    in_f,
                "param_count":    param_count,
                "w_bits":         w_bits,
                "a_bits":         a_bits if a_bits is not None else "None(wo)",
                "block_size":     block_size,
                # memory (actual = int8 storage, packed = theoretical bit-packing)
                "fp32_bytes":               fp32_bytes,
                "bf16_bytes":               bf16_bytes,
                "quant_bytes_actual":       quant_actual_b,
                "quant_bytes_packed":       quant_packed_b,
                "comp_vs_fp32_actual":      comp_fp32_actual,
                "comp_vs_bf16_actual":      comp_bf16_actual,
                "comp_vs_fp32_packed":      comp_fp32_packed,
                "comp_vs_bf16_packed":      comp_bf16_packed,
                # flops
                "base_flops":         base_flops,
                "w_flops_ratio":      round(w_ratio, 4),
                "a_flops_ratio":      round(a_ratio, 4),
                "wa_flops_ratio":     wa_ratio,
                "effective_flops_w":  int(eff_flops_w),
                "effective_flops_wa": int(eff_flops_wa),
            })

        # 합계 행
        rows.append({
            "layer_name":     "*** TOTAL ***",
            "component":      "-",
            "layer_type":     "-",
            "attn_mlp":       "-",
            "out_features":   "-",
            "in_features":    "-",
            "param_count":    sum(r["param_count"] for r in rows if isinstance(r["param_count"], int)),
            "w_bits":         "-",
            "a_bits":         "-",
            "block_size":     "-",
            "fp32_bytes":             total_fp32,
            "bf16_bytes":             total_bf16,
            "quant_bytes_actual":     total_quant_actual,
            "quant_bytes_packed":     total_quant_packed,
            "comp_vs_fp32_actual":    round(total_fp32 / total_quant_actual, 3),
            "comp_vs_bf16_actual":    round(total_bf16 / total_quant_actual, 3),
            "comp_vs_fp32_packed":    round(total_fp32 / total_quant_packed, 3),
            "comp_vs_bf16_packed":    round(total_bf16 / total_quant_packed, 3),
            "base_flops":             total_base_flops,
            "w_flops_ratio":          round(total_eff_flops_w  / total_base_flops, 4),
            "a_flops_ratio":          round(total_eff_flops_wa / total_eff_flops_w, 4) if total_eff_flops_w else "-",
            "wa_flops_ratio":         round(total_eff_flops_wa / total_base_flops, 4),
            "effective_flops_w":      int(total_eff_flops_w),
            "effective_flops_wa":     int(total_eff_flops_wa),
        })

        out_csv = output_dir / f"{exp_name}_layer_report.csv"
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n{'='*70}")
        print(f"  실험: {exp_name}")
        print(f"  success: {pc_success}%  |  layers: {len(rows)-1}")
        print(f"  메모리 (quantized linear layers만):")
        print(f"    FP32       : {total_fp32/1e6:.1f} MB")
        print(f"    BF16       : {total_bf16/1e6:.1f} MB")
        print(f"    Quant(int8): {total_quant_actual/1e6:.1f} MB  (현재 구현, int8 storage)")
        print(f"    Quant(pack): {total_quant_packed/1e6:.1f} MB  (이론 bit-packing)")
        print(f"    압축비 vs FP32: {total_fp32/total_quant_actual:.2f}x (actual) / {total_fp32/total_quant_packed:.2f}x (packed)")
        print(f"    압축비 vs BF16: {total_bf16/total_quant_actual:.2f}x (actual) / {total_bf16/total_quant_packed:.2f}x (packed)")
        print(f"  FLOPs (weight precision 기준):")
        print(f"    FP32 기준 FLOPs : {total_base_flops/1e9:.2f} GFLOPs")
        print(f"    Weight-only INT : {total_eff_flops_w/1e9:.2f} GFLOPs  ({total_eff_flops_w/total_base_flops*100:.1f}%)")
        print(f"    Weight+Act INT  : {total_eff_flops_wa/1e9:.2f} GFLOPs  ({total_eff_flops_wa/total_base_flops*100:.1f}%)")
        print(f"  저장: {out_csv}")

    # 실험 간 비교 요약
    print(f"\n{'='*90}")
    print("  실험별 비교 요약")
    print(f"  {'실험':<44} {'succ%':>6}  {'avgW':>5}  {'BF16_MB':>8}  {'actual_MB':>10}  {'packed_MB':>10}  {'comp_bf16_act':>14}  {'comp_bf16_pkg':>14}  {'eff_flops_wa%':>14}")
    print(f"  {'-'*140}")
    for result_path in args.result_jsons:
        p = Path(result_path)
        if not p.exists(): continue
        result = json.load(open(p))
        lc_all = result.get("layer_configs", {})
        pc = result.get("pc_success", "?")
        avg_w = sum(v["w_bits"] for v in lc_all.values()) / max(len(lc_all), 1)
        q_actual = 0; q_packed = 0; fp32_b = 0; bf16_b = 0
        base_f = 0; eff_w = 0; eff_wa = 0
        import math as _math
        for ln, lc in lc_all.items():
            ws = weight_stats.get(ln, {})
            sh = ws.get("shape")
            if not sh or len(sh) < 2: continue
            of, inf = int(sh[0]), int(sh[1])
            fp32_b += of*inf*4; bf16_b += of*inf*2
            qa, qp = bytes_for_quant_weight(of, inf, lc["w_bits"], lc["block_size"])
            q_actual += qa; q_packed += qp
            bf = 2*of*inf
            wr = lc["w_bits"]/32
            ar = (lc["a_bits"]/32) if lc["a_bits"] else 1.0
            base_f += bf; eff_w += bf*wr; eff_wa += bf*wr*ar
        print(f"  {p.stem:<44} {str(round(pc,1))+'%':>6}  {avg_w:>5.2f}  {bf16_b/1e6:>8.1f}  {q_actual/1e6:>10.1f}  {q_packed/1e6:>10.1f}  {bf16_b/q_actual:>14.2f}x  {bf16_b/q_packed:>14.2f}x  {eff_wa/base_f*100:>13.1f}%")


if __name__ == "__main__":
    main()
