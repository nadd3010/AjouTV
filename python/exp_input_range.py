#!/usr/bin/env python
"""
[실험] 입력 크기별 하드웨어 안정성 한계와 PC 쪽 클리핑 한계 (최종 가중치 + Q8.8 골든 모델 + Q4.4 입력)

질문: 슬라이더로 센서 값을 크게 움직이면(표준화 값이 커지면) Q8.8 산술이 넘쳐서(wrap) 하드웨어가
      수학적으로 의도한 판정(FP32 이상 모델)과 달라지는가? 어디까지가 안전한가? PC 에서 어디서 잘라야 하는가?

비교 기준(이상 모델 'ideal'): 같은 Q4.4 입력값을 FP32 로 계산한 점수. 하드웨어(Q8.8 골든 모델, RTL 수정 4가지 적용)와
  판정 일치율 / 점수 오차 / 중간값 최대(Q8.8 한계 128 에 얼마나 가까운가)를 본다.

[A] 센서별 반응 곡선: 정상 윈도우 하나의 센서 한 개를 4시점 내내 delta(표준편차 배수)만큼 이동 (슬라이더로 센서 하나를 움직이는 시연)
    -> 이상 판정 비율(ideal / 하드웨어), 판정 일치율, 점수 오차, 중간값 최대
[B] 센서별 안전 한계: 일치율 99% 이상이고 이상 모델의 중간값 최대가 120 미만인 가장 큰 |delta|
[C] 클리핑 한계 후보 L 별 최악 조건 점검: 모서리(네 센서가 모두 +-L 인 16가지), 무작위 내부(각 값이 [-L, L] 균일), 센서별 이동 + 클립
    -> 사전 선택 규칙: 세 점검 모두에서 일치율 >= 0.99 이고 중간값 최대 < 120 인 가장 큰 L
[D] 시연 가이드: 단일 센서를 +-delta 만큼 움직이면 하드웨어가 몇 %의 윈도우를 이상으로 판정하는가 (임계값 0.5)

[보완 점검 - 첫 실행 결과를 본 뒤 추가한 사후 기준이며, 위의 사전 규칙을 대체하지 않고 따로 보고한다]
  - 확실한 판정 뒤집힘(confident flip): 라벨이 다르고 이상 모델의 점수가 임계값에서 margin(기본 0.25) 이상 떨어진 윈도우의 비율.
    임계값 근처 점수가 반올림 때문에 뒤집히는 것(판정 일치 < 1)과 넘침으로 인한 오동작을 구분하기 위한 것.
  - 넘침 위험(risk): 이상 모델의 중간값 최대가 120 이상인 윈도우의 비율 (Q8.8 한계 128 에 대한 여유).
  [E] 시점 간 흔들림 허용치: 센서 값에 시점마다 독립 무작위 흔들림(+-eps)을 더했을 때
  [F] 램프 허용치: 센서 하나가 4시점 동안 R 만큼 변해 가는 입력(슬라이더를 빠르게 끄는 경우)

사용법
  python exp_input_range.py --weights res_square_out/model_fp32.npz --data dataset_t_hard --mem-dir ../mem --out range_out

필요: q88_golden.py, exp_input_encoding.py 를 같은 폴더에 둔다. numpy.
"""
import argparse
import csv
import json
import os

import numpy as np

import q88_golden as g
from exp_input_encoding import encode_to_q88, load_weights

SENSOR_NAMES = ["BEAMV", "IONP", "GASCH1", "COOLP"]
MX_LIMIT = 120.0          # Q8.8 한계 128 에 대한 여유


def evaluate(Xf, w, wi, roms, kw, thr, margin=0.25):
    """Xf: (N,4,4) 표준화 float -> Q4.4 반올림/포화 -> 이상 모델(FP32)과 하드웨어(Q8.8 골든) 비교."""
    xq, _ = encode_to_q88(np.clip(Xf, -8, 8), 8, 16, "round")
    xd = xq / 256.0
    z32, _, rng = g.fwd_fp32_w(w, xd)
    out = g.forward_q88(xq, wi["wq"], wi["wk"], wi["wv"], wi["wfc"], roms, **kw)
    hw = out["score"] / 256.0
    l32, lhw = z32 > thr, hw > thr
    pm = np.array([max(g.fwd_fp32_w(w, xd[i:i + 1])[2].values()) for i in range(len(xd))])      # 윈도우별 중간값 최대(이상 모델)
    return dict(ideal_rate=float(l32.mean()), hw_rate=float(lhw.mean()), agree=float((l32 == lhw).mean()),
                mae=float(np.abs(hw - z32).mean()), mx=float(max(rng.values())),
                wrap=float(max(out["stats"].values())) if out["stats"] else 0.0,
                conf_flip=float(((l32 != lhw) & (np.abs(z32 - thr) > margin)).mean()), risk=float((pm >= MX_LIMIT).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--mem-dir", required=True)
    ap.add_argument("--n-base", type=int, default=300, help="기준이 되는 정상 윈도우 수")
    ap.add_argument("--deltas", default="0.5,1,1.5,2,2.5,3,4,5,6,8")
    ap.add_argument("--clip-limits", default="3,4,5,6,8")
    ap.add_argument("--n-rand", type=int, default=3000)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.25, help="확실한 판정 뒤집힘으로 볼 임계값과의 거리")
    ap.add_argument("--jitters", default="0.05,0.1,0.25,0.5,1,2")
    ap.add_argument("--ramps", default="0.5,1,2,4,8")
    ap.add_argument("--out", default="range_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rs = np.random.default_rng(0)

    w = load_weights(args.weights)
    arch = str(w["arch"])
    wi = {k: g.to_q88(v) for k, v in w.items() if k in g.NUM_KEYS}
    roms = g.make_roms("repo", args.mem_dir)
    kw = dict(mul_bits=32, fc_signed=True, slice_start=3, inv_mode="full", arch=arch,
              b_int=int(wi["b"]) if "b" in wi else 0, c_int=wi.get("c"))
    te = np.load(os.path.join(args.data, "dataset_test.npz"))
    X, y = te["X"].astype(np.float64), te["y"]
    at = te["atype"] if "atype" in te.files else np.full(len(y), -1)
    pool = np.flatnonzero((y == 0) & (at < 0))
    base = X[rs.choice(pool, min(args.n_base, len(pool)), replace=False)]
    deltas = [float(v) for v in args.deltas.split(",")]
    Ls = [float(v) for v in args.clip_limits.split(",")]
    print(f"가중치 {args.weights} (arch={arch}), 기준 정상 윈도우 {len(base)}개, 입력은 Q4.4(+-8)로 인코딩, 임계값 {args.thr}")
    base_ev = evaluate(base, w, wi, roms, kw, args.thr, args.margin)
    print(f"기준(이동 없음): 이상 판정 {base_ev['hw_rate']:.1%} (정상 윈도우이므로 오경보율), 중간값 최대 {base_ev['mx']:.1f}")

    # ---------------- [A] 센서별 반응 곡선 ----------------
    rows = []
    curve = {}
    for s in range(4):
        for sign in (-1, 1):
            for d in deltas:
                Xs = base.copy()
                Xs[:, s, :] += sign * d
                ev = evaluate(Xs, w, wi, roms, kw, args.thr, args.margin)
                curve[(s, sign, d)] = ev
                rows.append(dict(sensor=SENSOR_NAMES[s], delta=sign * d, **ev))
    print("\n=== [A] 센서 하나를 이동했을 때 (정상 윈도우 기준, 4시점 내내 이동) ===")
    for s in range(4):
        print(f"\n  {SENSOR_NAMES[s]}")
        print(f"  {'delta(σ)':>9}{'ideal 이상%':>13}{'HW 이상%':>10}{'판정일치':>9}{'점수오차':>9}{'중간값최대':>10}{'wrap':>8}")
        for sign in (-1, 1):
            for d in (deltas if sign > 0 else deltas[::-1]):
                ev = curve[(s, sign, d)]
                print(f"  {sign * d:>+9.1f}{ev['ideal_rate']:>13.1%}{ev['hw_rate']:>10.1%}{ev['agree']:>9.3f}{ev['mae']:>9.3f}{ev['mx']:>10.1f}{ev['wrap']:>8.3f}")
    with open(os.path.join(args.out, "demo_curve.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # ---------------- [B] 센서별 안전 한계 ----------------
    print("\n=== [B] 센서별 안전 한계 (판정 일치 >= 0.99 이고 이상 모델의 중간값 최대 < %.0f 인 가장 큰 |delta|) ===" % MX_LIMIT)
    safe = {}
    for s in range(4):
        for sign in (-1, 1):
            last = 0.0
            for d in deltas:
                ev = curve[(s, sign, d)]
                if ev["agree"] >= 0.99 and ev["mx"] < MX_LIMIT:
                    last = d
                else:
                    break
            safe[(s, sign)] = last
        print(f"  {SENSOR_NAMES[s]:<8} 음(-) 방향 {safe[(s, -1)]:>4.1f}σ, 양(+) 방향 {safe[(s, 1)]:>4.1f}σ")

    # ---------------- [C] 클리핑 한계 후보별 최악 조건 점검 ----------------
    print("\n=== [C] PC 쪽 클리핑 한계 L (표준화 값을 +-L 로 자른 뒤 Q4.4 로 전송) ===")
    print(f"{'L(σ)':>6}{'모서리 일치':>12}{'모서리 mx':>11}{'내부 일치':>10}{'내부 mx':>9}{'이동+클립 일치':>15}{'이동+클립 mx':>13}{'L 에서 단일센서 감지율':>22}{'판정':>8}")
    results_c = {}
    for L in Ls:
        corners = np.array(np.meshgrid(*[[-L, L]] * 4)).reshape(4, -1).T                    # (16,4)
        Xc = np.repeat(corners[:, :, None], 4, axis=2)                                      # (16,4,4)
        ec = evaluate(Xc, w, wi, roms, kw, args.thr, args.margin)
        Xr = rs.uniform(-L, L, size=(args.n_rand, 4, 4))
        er = evaluate(Xr, w, wi, roms, kw, args.thr, args.margin)
        ag, mxs, sens, cfs, rks = [], [], [], [ec["conf_flip"]], [ec["risk"]]
        for s in range(4):
            for d in deltas + [10.0]:
                for sign in (-1, 1):
                    Xs = base.copy()
                    Xs[:, s, :] += sign * d
                    ev = evaluate(np.clip(Xs, -L, L), w, wi, roms, kw, args.thr, args.margin)
                    ag.append(ev["agree"])
                    mxs.append(ev["mx"])
                    cfs.append(ev["conf_flip"])
                    rks.append(ev["risk"])
            for sign in (-1, 1):
                Xs = base.copy()
                Xs[:, s, :] = sign * L
                sens.append(evaluate(Xs, w, wi, roms, kw, args.thr, args.margin)["hw_rate"])
        ok = ec["agree"] >= 0.99 and er["agree"] >= 0.99 and min(ag) >= 0.99 and max(ec["mx"], er["mx"], max(mxs)) < MX_LIMIT
        results_c[L] = dict(corner_agree=ec["agree"], corner_mx=ec["mx"], rand_agree=er["agree"], rand_mx=er["mx"],
                            sweep_agree=float(min(ag)), sweep_mx=float(max(mxs)), single_sensor_detect=float(np.mean(sens)), ok=bool(ok),
                            rand_conf_flip=er["conf_flip"], rand_risk=er["risk"], tc_conf_flip=float(max(cfs)), tc_risk=float(max(rks)),
                            worst_detect=float(min(sens)))
        r = results_c[L]
        print(f"{L:>6.1f}{r['corner_agree']:>12.3f}{r['corner_mx']:>11.1f}{r['rand_agree']:>10.3f}{r['rand_mx']:>9.1f}"
              f"{r['sweep_agree']:>15.3f}{r['sweep_mx']:>13.1f}{r['single_sensor_detect']:>22.1%}{'충족' if ok else '미충족':>8}")
    okL = [L for L, r in results_c.items() if r["ok"]]
    choice = max(okL) if okL else None
    print("\n=== 사전 선택 규칙 적용 (세 점검 모두 일치 >= 0.99 이고 중간값 최대 < %.0f 인 가장 큰 L) ===" % MX_LIMIT)
    if choice is not None:
        print(f"  -> 선택: L = {choice:g}σ  (sensor_encoder.py 의 CLIP_Z 를 {choice:g} 로)")
    else:
        print("  -> 조건을 만족하는 L 이 없습니다. 후보 중 가장 작은 L 의 결과를 보고 --clip-limits 에 더 작은 값을 넣어 보세요.")

    # ---------------- 보완 점검 (사후 기준) ----------------
    print("\n=== 보완 점검(사후 기준): 시간에 따라 변하지 않는 입력(슬라이더로 센서 값을 고정)에 대한 안정성 ===")
    print(f"{'L(σ)':>6}{'확실한 뒤집힘(최대)':>20}{'넘침 위험(최대)':>16}{'L 에서 최악 센서 감지율':>24}{'안정':>6}")
    tc_ok = {}
    for L, r in results_c.items():
        tc_ok[L] = r["tc_conf_flip"] <= 0.01 and r["tc_risk"] == 0.0
        print(f"{L:>6.1f}{r['tc_conf_flip']:>20.3%}{r['tc_risk']:>16.3%}{r['worst_detect']:>24.1%}{'O' if tc_ok[L] else 'X':>6}")
    cand = [L for L in Ls if tc_ok[L]]
    choice2 = max(cand, key=lambda L: (results_c[L]["worst_detect"], -L)) if cand else None
    print("  보완 선택 기준: 안정(O)인 L 중 'L 에서 최악 센서 감지율'이 가장 높은 값 (동률이면 작은 L)")
    print(f"  -> 보완 기준 선택: {('L = %gσ' % choice2) if choice2 is not None else '없음'}")
    natural = float(np.median(np.std(base, axis=2)))
    print(f"\n  참고: 기준 정상 윈도우의 시점 간 변동(센서별 4시점 표준편차의 중앙값) = {natural:.3f}σ")
    Lref = choice2 if choice2 is not None else (min(Ls) if Ls else 4.0)
    print(f"\n=== [E] 시점 간 흔들림 허용치 (L = {Lref:g}σ 로 자른 입력에 시점마다 독립 +-eps 무작위 흔들림을 더함) ===")
    print(f"{'eps(σ)':>8}{'확실한 뒤집힘(최대)':>20}{'넘침 위험(최대)':>16}{'중간값최대(최대)':>16}")
    jit_ok = None
    jitter_res = {}
    for e in [float(v) for v in args.jitters.split(",")]:
        cf, rk, mxj = [], [], []
        conds = [(None, 0)] + [(s2, sg) for s2 in range(4) for sg in (-1, 1)]
        for (s2, sg) in conds:
            Xs = base.copy()
            if s2 is not None:
                Xs[:, s2, :] += sg * Lref
            Xs = np.clip(Xs + rs.uniform(-e, e, size=Xs.shape), -Lref, Lref)
            ev = evaluate(Xs, w, wi, roms, kw, args.thr, args.margin)
            cf.append(ev["conf_flip"]); rk.append(ev["risk"]); mxj.append(ev["mx"])
        jitter_res[e] = dict(conf_flip=float(max(cf)), risk=float(max(rk)), mx=float(max(mxj)))
        ok_e = max(cf) <= 0.01 and max(rk) <= 0.01
        if ok_e:
            jit_ok = e
        print(f"{e:>8.2f}{max(cf):>20.3%}{max(rk):>16.3%}{max(mxj):>16.1f}{'  <- 허용' if ok_e else ''}")
    print(f"  -> 허용되는 최대 흔들림: {('+-%gσ' % jit_ok) if jit_ok is not None else '가장 작은 후보에서도 불안정'}  (기준: 확실한 뒤집힘 1% 이하, 넘침 위험 1% 이하)")
    print(f"\n=== [F] 램프 허용치 (센서 하나가 4시점 동안 R 만큼 변해 감, L = {Lref:g}σ 로 클립) ===")
    print(f"{'R(σ)':>8}{'확실한 뒤집힘(최대)':>20}{'넘침 위험(최대)':>16}{'중간값최대(최대)':>16}")
    ramp_ok = None
    ramp_res = {}
    for R in [float(v) for v in args.ramps.split(",")]:
        cf, rk, mxr = [], [], []
        for s2 in range(4):
            for sg in (-1, 1):
                Xs = base.copy()
                Xs[:, s2, :] += sg * R * np.arange(4)[None, :] / 3.0
                ev = evaluate(np.clip(Xs, -Lref, Lref), w, wi, roms, kw, args.thr, args.margin)
                cf.append(ev["conf_flip"]); rk.append(ev["risk"]); mxr.append(ev["mx"])
        ramp_res[R] = dict(conf_flip=float(max(cf)), risk=float(max(rk)), mx=float(max(mxr)))
        ok_r = max(cf) <= 0.01 and max(rk) <= 0.01
        if ok_r:
            ramp_ok = R
        print(f"{R:>8.2f}{max(cf):>20.3%}{max(rk):>16.3%}{max(mxr):>16.1f}{'  <- 허용' if ok_r else ''}")
    print(f"  -> 4시점(약 8 ms) 동안 허용되는 최대 변화: {('%gσ' % ramp_ok) if ramp_ok is not None else '가장 작은 후보에서도 불안정'}")

    # ---------------- [D] 시연 가이드 ----------------
    print("\n=== [D] 시연 가이드: 정상 윈도우의 센서 하나를 이동했을 때 하드웨어가 이상으로 판정하는 윈도우 비율 ===")
    print(f"  {'센서':<8}{'방향':<4}" + "".join(f"{d:>7.1f}" for d in deltas))
    for s in range(4):
        for sign in (1, -1):
            print(f"  {SENSOR_NAMES[s]:<8}{('+' if sign > 0 else '-'):<4}" + "".join(f"{curve[(s, sign, d)]['hw_rate']:>7.0%}" for d in deltas))
    g50 = {}
    for s in range(4):
        for sign in (-1, 1):
            hit = [d for d in deltas if curve[(s, sign, d)]["hw_rate"] >= 0.5]
            g50[(SENSOR_NAMES[s], sign)] = hit[0] if hit else None
    print("  (이상 판정 비율이 50% 에 처음 도달하는 이동 크기: " +
          ", ".join(f"{k[0]}{'+' if k[1] > 0 else '-'} {('%.1fσ' % v) if v is not None else '8σ 이내 없음'}" for k, v in g50.items()) + ")")

    with open(os.path.join(args.out, "range_results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), base=base_ev, safe={f"{SENSOR_NAMES[k[0]]}{'+' if k[1] > 0 else '-'}": v for k, v in safe.items()},
                       clip=results_c, choice=choice, choice_supplementary=choice2, jitter=jitter_res, ramp=ramp_res), f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장: {args.out}/range_results.json, demo_curve.csv")


if __name__ == "__main__":
    main()