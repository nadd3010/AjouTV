#!/usr/bin/env python
"""
[검증] 시연 시나리오를 PC 인코더(클리핑 + 기울기 제한)와 하드웨어 골든 모델에 흘려 안정성을 확인한다.

시나리오 (표준화된 값, 시점 하나 = 센서 4바이트 한 묶음). 기준 상태는 정상 윈도우의 센서별 값.
  점프    : 센서 하나를 갑자기 +-8σ 목표로 바꿈 (클리핑 후 +-L). 8시점 유지 -> 점프 -> 목표 유지.  (슬라이더를 확 끄는 경우)
  모서리  : 네 센서를 한꺼번에 +-L 패턴(+/+/+/+, -/-/-/-, +/-/+/-, -/+/-/+)으로 점프.
  스윕    : 센서 하나를 -L 에서 +L 까지 8시점(빠름) 또는 40시점(느림)에 걸쳐 이동.
방식 두 가지를 같은 시나리오로 비교한다.
  (A) 클리핑만   : 값을 +-L 로 자르고 그대로 전송 (기울기 제한 없음)
  (B) 클리핑 + 기울기 제한 : sensor_encoder.prepare_stream (시점마다 변화 <= slew)
지표(스트림의 모든 슬라이딩 윈도우 기준)
  넘침 위험      : 이상 모델의 중간값 최대가 120 이상인 윈도우 비율
  확실한 뒤집힘  : 하드웨어 판정이 이상 모델과 다르고, 이상 모델 점수가 임계값에서 margin 이상 떨어진 윈도우 비율
  최종 감지율    : 마지막 4개 윈도우(목표 상태가 4시점 모두 채워진 뒤)에서 하드웨어가 이상으로 판정한 비율 (점프/모서리 시나리오)

사용법
  python exp_demo_sequences.py --weights res_square_out/model_fp32.npz --data dataset_t_hard --mem-dir ../mem

필요: q88_golden.py, exp_input_encoding.py, exp_input_range.py, sensor_encoder.py 를 같은 폴더에 둔다.
"""
import argparse
import json
import os

import numpy as np

import q88_golden as g
import sensor_encoder as se
from exp_input_encoding import load_weights
from exp_input_range import evaluate


def windows_from_stream(z):
    """z: (4,T) -> (T-3, 4, 4) 슬라이딩 윈도우 (window[w][센서][시점] = z[센서, w+시점])."""
    T = z.shape[1]
    return np.stack([z[:, k:k + 4] for k in range(T - 3)])


def build_sequences(base_vec, L, hold, slew, target=8.0):
    """기준 값(4,) 에서 시작하는 시나리오 스트림들 (이름, 스트림(4,T), 마지막이 목표 상태로 유지되는지).
    기울기 제한이 있어도 목표에 충분히 도달하도록 뒤쪽 유지 구간을 ceil(2L/slew)+4 시점 더 길게 잡는다 (두 방식에 같은 시나리오를 쓴다)."""
    post = hold + int(np.ceil(2 * L / max(slew, 1e-6))) + 4
    seqs = []
    for s in range(4):
        for sg in (-1, 1):
            z = np.tile(base_vec[:, None], (1, hold + post))
            z[s, hold:] = base_vec[s] + sg * target
            seqs.append((f"점프 {('+' if sg > 0 else '-')}{se.SENSORS[s][:5]}", z, True))
    for pat in ((1, 1, 1, 1), (-1, -1, -1, -1), (1, -1, 1, -1), (-1, 1, -1, 1)):
        z = np.tile(base_vec[:, None], (1, hold + post))
        z[:, hold:] = (np.array(pat) * target)[:, None]
        seqs.append((f"모서리 {''.join('+' if v > 0 else '-' for v in pat)}", z, True))
    for s in range(4):
        for n in (8, 40):
            z = np.tile(base_vec[:, None], (1, hold + n + post))
            z[s, hold:hold + n] = np.linspace(-target, target, n)
            z[s, hold + n:] = target
            seqs.append((f"스윕 {se.SENSORS[s][:5]} {n}시점", z, False))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--mem-dir", required=True)
    ap.add_argument("--clip", type=float, default=se.CLIP_Z)
    ap.add_argument("--slew", type=float, default=se.SLEW_STEP)
    ap.add_argument("--n-base", type=int, default=20)
    ap.add_argument("--hold", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rs = np.random.default_rng(0)

    w = load_weights(args.weights)
    wi = {k: g.to_q88(v) for k, v in w.items() if k in g.NUM_KEYS}
    roms = g.make_roms("repo", args.mem_dir)
    kw = dict(mul_bits=32, fc_signed=True, slice_start=3, inv_mode="full", arch=str(w["arch"]),
              b_int=int(wi["b"]) if "b" in wi else 0, c_int=wi.get("c"))
    te = np.load(os.path.join(args.data, "dataset_test.npz"))
    X, y = te["X"].astype(np.float64), te["y"]
    at = te["atype"] if "atype" in te.files else np.full(len(y), -1)
    pool = np.flatnonzero((y == 0) & (at < 0))
    bases = X[rs.choice(pool, min(args.n_base, len(pool)), replace=False)].mean(axis=2)          # (n,4) 센서별 기준 값
    print(f"가중치 {args.weights}, 기준 상태 {len(bases)}개, 클리핑 L={args.clip:g}σ, 기울기 제한 {args.slew:g}σ/시점, 임계값 {args.thr}")

    results = {}
    kinds = sorted({n.split()[0] for n, _, _ in build_sequences(bases[0], args.clip, args.hold, args.slew)})
    for variant in ("A 클리핑만", "B 클리핑+기울기 제한"):
        allw, finalw, per_kind = [], [], {k: [] for k in kinds}
        for b in bases:
            for name, z, final in build_sequences(b, args.clip, args.hold, args.slew):
                z = np.clip(z, -args.clip, args.clip)
                if variant.startswith("B"):
                    z = se.slew_limit(z, args.slew)
                wins = windows_from_stream(z)
                allw.append(wins)
                per_kind[name.split()[0]].append(wins)
                if final:
                    finalw.append(wins[-4:])
        res = {}
        ev = evaluate(np.concatenate(allw), w, wi, roms, kw, args.thr, args.margin)
        ev_final = evaluate(np.concatenate(finalw), w, wi, roms, kw, args.thr, args.margin)
        res["all"] = dict(windows=int(sum(len(a) for a in allw)), risk=ev["risk"], conf_flip=ev["conf_flip"], mx=ev["mx"],
                          final_detect=ev_final["hw_rate"], final_ideal=ev_final["ideal_rate"])
        for k, lst in per_kind.items():
            e = evaluate(np.concatenate(lst), w, wi, roms, kw, args.thr, args.margin)
            res[k] = dict(windows=int(sum(len(a) for a in lst)), risk=e["risk"], conf_flip=e["conf_flip"], mx=e["mx"])
        results[variant] = res

    print("\n=== 시연 시나리오 안정성 (스트림의 모든 슬라이딩 윈도우) ===")
    print(f"{'방식':<22}{'시나리오':<10}{'윈도우':>9}{'넘침 위험':>11}{'확실한 뒤집힘':>14}{'중간값최대':>11}")
    for variant, res in results.items():
        for k in ["all"] + kinds:
            r = res[k]
            print(f"{variant:<22}{('전체' if k == 'all' else k):<10}{r['windows']:>9}{r['risk']:>11.2%}{r['conf_flip']:>14.2%}{r['mx']:>11.1f}")
    print("\n=== 목표 상태 도달 후 감지율 (점프/모서리 시나리오의 마지막 4개 윈도우) ===")
    for variant, res in results.items():
        r = res["all"]
        print(f"  {variant:<22} 하드웨어 이상 판정 {r['final_detect']:.1%} (이상 모델 {r['final_ideal']:.1%})")
    a, b = results["A 클리핑만"]["all"], results["B 클리핑+기울기 제한"]["all"]
    ok_b = b["risk"] <= 0.01 and b["conf_flip"] <= 0.01
    print("\n=== 판정 (기준: 넘침 위험 1% 이하, 확실한 뒤집힘 1% 이하) ===")
    print(f"  A 클리핑만            : 넘침 {a['risk']:.2%}, 뒤집힘 {a['conf_flip']:.2%} -> {'충족' if a['risk'] <= 0.01 and a['conf_flip'] <= 0.01 else '미충족'}")
    print(f"  B 클리핑+기울기 제한  : 넘침 {b['risk']:.2%}, 뒤집힘 {b['conf_flip']:.2%} -> {'충족' if ok_b else '미충족'}")
    print("  해석: B 가 충족이면 PC 인코더(sensor_encoder.prepare_stream)를 쓰는 한 시연 입력에서 하드웨어가 안정적이다.")
    print("        B 가 미충족이면 --slew 를 줄이거나(예: 0.25) --clip 을 낮춰 다시 확인한다.")
    with open(os.path.join(args.out, "demo_sequences.json"), "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), results=results), f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장: {args.out}/demo_sequences.json")


if __name__ == "__main__":
    main()