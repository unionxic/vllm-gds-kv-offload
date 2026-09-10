#!/usr/bin/env python3
"""OPT-66B 계열 결과 json을 한 표로 모으고, forward 비용을 고정비 모형과 대조한다.

모형: forward_s = host_tier_gib*GiB/H2D_GBPS + ssd_tier_gib*GiB/SSD_GBPS
  H2D_GBPS  12.3  pinned host -> GPU 실측 (PCIe 3.0 x16, pin_exact_test.py)
  SSD_GBPS   3.44 970 EVO cuFile 실측 3.2 GiB/s (08c gdsio, 06 decode 중 NVMe 2.96~3.3 GB/s)

새 결과가 나오면 먼저 이 표를 보고, 모형과 15% 이상 어긋난 런(!)과 기준 런 대비
10% 이상 움직인 지표(*)를 설명한 뒤에 다른 원인을 찾는다.

사용:
  python tools/compare_results.py                 # 기본 경로 전부
  python tools/compare_results.py --ref ab-none   # 기준 런 대비 변화율
  python tools/compare_results.py results/kv-policy/ab-*.json
"""
import argparse
import glob
import json
import os
import sys
import time

GIB = 1024 ** 3
H2D_GBPS = 12.3e9
SSD_GBPS = 3.44e9
DEFAULT_DIRS = [
    "results/weight-offload/opt66b",
    "results/combined/opt66b",
    "results/kv-policy",
    "results/model-host-baseline",
]


def model_forward_s(tiers):
    if not tiers:
        return None
    h = tiers.get("host_tier_gib") or 0.0
    s = tiers.get("ssd_tier_gib") or 0.0
    return h * GIB / H2D_GBPS + s * GIB / SSD_GBPS


def extract(path):
    with open(path) as f:
        d = json.load(f)
    a = d.get("args", {})
    t = d.get("tiers", {})
    row = {
        "tag": os.path.basename(path)[:-5],
        "wt": (a.get("weight_transport") or a.get("transport") or "-")[:6],
        "date": time.strftime("%m-%d", time.localtime(os.path.getmtime(path))),
        "host": a.get("host_fraction"),
        "res": t.get("gpu_resident_layers"),
        "host_gib": t.get("host_tier_gib"),
        "ssd_gib": t.get("ssd_tier_gib"),
        "kvt": a.get("kv_transport", "-"),
        "kv_gib": a.get("kv_cache_gib"),
        "np": a.get("n_prompts") or a.get("batch"),
        "wall": None,
        "fwd_n": None,
        "fwd_meas": None,
        "kv_r_gib": None,
        "kv_w_gib": None,
        "path": path,
    }
    row["fwd_model"] = model_forward_s(t)

    if "rounds" in d:  # run_phase_66b.py: step 단위 계측
        rs = d["rounds"]
        row["wall"] = sum(r.get("wall_s", 0) for r in rs)
        # 폴링 step(토큰 0)은 forward가 아니므로 제외
        row["fwd_n"] = sum(
            (sum(1 for st in r["steps"] if st["t1"] - st["t0"] > 0.5) if r.get("steps")
             else r.get("n_prefill_steps", 0) + r.get("n_decode_steps", 0)) for r in rs)
        # 러너의 phase 라벨은 첫 토큰 대기 중인 요청이 있으면 decode forward도 prefill로 적으므로 쓰지 않는다.
        # 출력이 있는 step(decode forward가 대부분)의 중앙값. prefill forward는 출력 0으로 기록되어 제외됨.
        durs = sorted(st["t1"] - st["t0"] for r in rs for st in (r.get("steps") or [])
                      if st.get("n_out", 0) > 0 and st["t1"] - st["t0"] > 0.5)
        if durs:
            row["fwd_meas"] = durs[len(durs) // 2]
        else:
            dec_w = sum(r["decode_steps"]["wall_s"] for r in rs if r.get("decode_steps"))
            dec_n = sum(r.get("n_decode_steps", 0) for r in rs)
            row["fwd_meas"] = dec_w / dec_n if dec_n else None
        row["kv_r_gib"] = sum(r["kv_io"]["read_gib"] for r in rs if r.get("kv_io"))
        row["kv_w_gib"] = sum(r["kv_io"]["write_gib"] for r in rs if r.get("kv_io"))
    elif "r1" in d:  # run_combo_66b.py / run_policy_66b.py: generate 두 번 방식
        rs = [d[k] for k in ("r1", "r2") if k in d]
        row["wall"] = sum(r.get("gen_total_s", 0) for r in rs)
        # kv 1.5 GiB 구성은 batch 1 순차라 decode_step_s가 n_prompts개 forward의 합
        seq = (a.get("kv_cache_gib") or 99) < 2.0
        div = (a.get("n_prompts") or 1) if seq else None
        if div:
            row["fwd_meas"] = sum(r["decode_step_s"] for r in rs) / len(rs) / div
        row["kv_r_gib"] = sum(r.get("kv_read_b", 0) for r in rs) / GIB
        row["kv_w_gib"] = sum(r.get("kv_write_b", 0) for r in rs) / GIB
    elif "decode_step_s" in d:  # run_66b.py: 가중치 경로만
        row["wall"] = d.get("gen_total_s")
        row["fwd_meas"] = d.get("decode_step_s")
    return row


def fmt(v, w, p=1):
    if v is None:
        return "-".rjust(w)
    if isinstance(v, float):
        return f"{v:.{p}f}".rjust(w)
    return str(v).rjust(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--ref", help="기준 런 tag. 지정 시 wall과 forward의 변화율을 덧붙임")
    ap.add_argument("--tol-model", type=float, default=0.15)
    ap.add_argument("--tol-ref", type=float, default=0.10)
    ap.add_argument("--sort", choices=["date", "tag"], default="date")
    ap.add_argument("--all-configs", action="store_true",
                    help="기준 런과 구성(host, 상주, KV, 프롬프트 수)이 달라도 변화율을 표시")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = args.paths or [p for dd in DEFAULT_DIRS for p in glob.glob(os.path.join(root, dd, "*.json"))]
    rows = []
    for p in sorted(paths):
        try:
            rows.append(extract(p))
        except Exception as e:  # 형식이 다른 json은 건너뜀
            print(f"skip {p}: {e}", file=sys.stderr)
    if args.sort == "date":
        rows.sort(key=lambda r: os.path.getmtime(r["path"]))
    else:
        rows.sort(key=lambda r: r["tag"])

    ref = next((r for r in rows if r["tag"] == args.ref), None) if args.ref else None
    if args.ref and not ref:
        print(f"기준 런 {args.ref} 없음", file=sys.stderr)

    def cfg(r):
        return (r["host"], r["res"], r["kv_gib"], r["np"])

    hdr = (f"{'date':5} {'tag':24} {'wt':6} {'host':5} {'res':3} {'hostGiB':8} {'ssdGiB':7} {'kvt':11} {'kvGiB':5} "
           f"{'np':3} {'wall_s':8} {'fwd_n':5} {'fwd_s':6} {'model':6} {'dev%':6} {'kvR':6} {'kvW':6}")
    if ref:
        hdr += f" {'dWall%':7} {'dFwd%':6}"
    print(hdr)
    print("-" * len(hdr))
    flagged = []
    for r in rows:
        dev = None
        if r["fwd_meas"] and r["fwd_model"]:
            dev = (r["fwd_meas"] - r["fwd_model"]) / r["fwd_model"] * 100
        # 모형은 cuFile 경로 기준. POSIX 가중치 경로는 2.4배 느린 것이 정상이라 표시만 하고 플래그 없음
        mark = "!" if (dev is not None and abs(dev) > args.tol_model * 100 and r["wt"] == "cufile") else " "
        line = (f"{r['date']:5} {r['tag'][:24]:24} {r['wt']:6} {fmt(r['host'],5,2)} {fmt(r['res'],3)} {fmt(r['host_gib'],8)} "
                f"{fmt(r['ssd_gib'],7)} {str(r['kvt'])[:11]:11} {fmt(r['kv_gib'],5)} {fmt(r['np'],3)} "
                f"{fmt(r['wall'],8)} {fmt(r['fwd_n'],5)} {fmt(r['fwd_meas'],6)} {fmt(r['fwd_model'],6)} "
                f"{fmt(dev,5)}{mark} {fmt(r['kv_r_gib'],6)} {fmt(r['kv_w_gib'],6)}")
        if ref and (args.all_configs or cfg(r) == cfg(ref)):
            dw = df = None
            if r["wall"] and ref["wall"]:
                dw = (r["wall"] - ref["wall"]) / ref["wall"] * 100
            if r["fwd_meas"] and ref["fwd_meas"]:
                df = (r["fwd_meas"] - ref["fwd_meas"]) / ref["fwd_meas"] * 100
            m = "*" if (dw is not None and abs(dw) > args.tol_ref * 100) else " "
            line += f" {fmt(dw,6)}{m} {fmt(df,6)}"
            if m == "*":
                flagged.append((r["tag"], "기준 대비 wall", dw))
        elif ref:
            line += f" {'':7} {'':6}"
        if mark == "!":
            flagged.append((r["tag"], "모형 대비 forward", dev))
        print(line)

    print()
    print(f"모형: host_gib/{H2D_GBPS/1e9:.1f} GB/s + ssd_gib/{SSD_GBPS/1e9:.2f} GB/s. "
          f"fwd_s는 decode step 하나(가중치 전송 고정비). 07·정책 러너는 batch 1일 때만 산출. "
          f"POSIX 가중치 경로는 모형 대상 아님.")
    if flagged:
        print("설명이 필요한 런:")
        for tag, why, v in flagged:
            print(f"  {tag}: {why} {v:+.0f}%")


if __name__ == "__main__":
    main()
