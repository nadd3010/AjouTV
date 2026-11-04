#!/usr/bin/env python
"""
[실험] 관계 이탈(corr_break) 이상 + 어려운 음성 데이터셋에서 구조별 비교 (통제 실험)

전제: make_dataset_corr.py 로 만든 데이터셋. exp_head.py 와 make_dataset_corr.py 를 같은 폴더에 둔다.

비교 대상
  기준선(학습 없음 또는 가벼운 모델)
    센서별 임계값(max|z|)        계획서의 Threshold 방식을 가장 관대하게 흉내: 어느 한 값이라도 크면 이상
    엄격한 범위 점검             관측 범위를 벗어난 값이 있으면 이상 (합성 이상은 항상 범위 안이라 AUC 0.5 가 되어야 함)
    마할라노비스 16차원          정상 윈도우만으로 피팅하는 비지도 기준 (공분산을 아는 탐지기)
    비선형 baseline(HGB)         지도 학습 기준
  신경망 (exp_head.py 의 구조)
    rtl                    기본 RTL 구조 (attention + 선형 FC, 바이어스 없음)
    res_square             attention + 잔차 + 제곱 머리 (우리 구조)
    uniform_res_square     attention 가중치를 1/4 로 고정한 대조군 (센서 평균 문맥만 남김)
    raw_square             attention 없이 X^2 . w + b (센서 간 섞기 없음)

사전 판정 기준 (결과를 보기 전에 정해 둔 것, 마지막에 자동 출력)
  A. res_square 전체 AUC 가 센서별 임계값 AUC 보다 0.20 이상 높은가
  B. res_square 전체 AUC 가 참고 상한(마할라노비스, HGB 중 높은 쪽)의 90% 이상인가
  C. res_square 와 uniform_res_square 의 전체 AUC 차이가 0.03 미만인가 (-> attention 의 이득은 '평균 문맥')
  D. (참고) 무해한 큰 변동 음성만 상대로 한 AUC 가 0.75 이상인가 (큰 이동과 소축 이탈을 구분하는가)

사용법
  python exp_corr_break.py --data dataset_corr --out corr_out
  python exp_corr_break.py --data dataset_corr --out corr_out --models rtl,res_square   (일부만)
  python exp_corr_break.py --data dataset_corr --baselines-only                         (torch 없이 기준선만)

필요 패키지: numpy, scikit-learn, torch(신경망 모델에만)
"""
import argparse
import json
import os
import time

import numpy as np

import exp_head as eh
import make_dataset_corr as mdc

DEFAULT_MODELS = ["rtl", "res_square", "uniform_res_square", "raw_square"]


def load_split(d, split):
    z = np.load(os.path.join(d, f"dataset_{split}.npz"))
    return {k: z[k] for k in z.files}


def fmt(r):
    return (f"{r['all']:>8.3f}{r['plain']:>9.3f}{r['benign']:>10.3f}{r['d2-3']:>8.3f}{r['d3-4']:>8.3f}{r['d4-5']:>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_corr")
    ap.add_argument("--out", default="corr_out")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--hw-penalty", type=float, default=0.1)
    ap.add_argument("--hw-limit", type=float, default=64.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tr, va, te = load_split(args.data, "train"), load_split(args.data, "val"), load_split(args.data, "test")
    n_b = int((te["atype"] == -2).sum())
    print(f"학습 {len(tr['y']):,} / 검증 {len(va['y']):,} / 평가 {len(te['y']):,} (평가 중 이상 {int(te['y'].sum()):,}, 무해한 큰 변동 음성 {n_b:,})")

    rows = {}
    sc = mdc.baseline_scores(tr, te)
    for nm, key in (("센서별 임계값(max|z|)", "maxz"), ("엄격한 범위 점검", "range"), ("마할라노비스 16차원", "maha")):
        rows[nm] = dict(res=mdc.report_scores(te, sc[key]), params="-", maxabs="-")
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        def feat(X):
            n = len(X)
            pair = [X[:, i, :].mean(axis=1) - X[:, j, :].mean(axis=1) for i in range(4) for j in range(i + 1, 4)]
            return np.c_[X.reshape(n, -1), np.stack(pair, axis=1)]
        m = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(feat(tr["X"]), tr["y"])
        rows["비선형 baseline(HGB)"] = dict(res=mdc.report_scores(te, m.predict_proba(feat(te["X"]))[:, 1]), params="-", maxabs="-")
    except Exception as e:  # noqa
        print("HGB 생략:", e)

    nn_res = {}
    if not args.baselines_only:
        names = [m.strip() for m in args.models.split(",")]
        for nm in names:
            if nm not in eh.SPECS:
                raise SystemExit(f"알 수 없는 구조 {nm}. 가능: {', '.join(eh.SPECS)}")
        for nm in names:
            spec, t0, best = eh.SPECS[nm], time.time(), (-1.0, None)
            for r in range(max(1, args.restarts)):
                model, v_auc, ep = eh.train_one(spec, tr, va, args, args.seed + r)
                print(f"  [{nm}] 시작값 {r + 1}/{args.restarts}: val AUC {v_auc:.3f} (epoch {ep})", flush=True)
                if v_auc > best[0]:
                    best = (v_auc, model)
            z, mx = eh.predict(best[1], te["X"])
            res = mdc.report_scores(te, z)
            params, _ = eh.cost(spec)
            rows[f"{nm}"] = dict(res=res, params=params, maxabs=mx, label=spec["label"])
            nn_res[nm] = res
            print(f"[{nm}] Test AUC {res['all']:.3f} (일반 {res['plain']:.3f}, 무해 {res['benign']:.3f})  중간값 최대 {mx:.1f}  {time.time() - t0:.0f}초", flush=True)

    print("\n=== 결과 (Test AUC: 이상 vs 음성) ===")
    print(f"{'구성':<30}{'전체':>8}{'일반음성':>9}{'무해변동':>10}{'d2-3':>8}{'d3-4':>8}{'d4-5':>8}{'파라미터':>9}{'중간값최대':>10}")
    for nm, r in rows.items():
        mx = f"{r['maxabs']:.1f}" if not isinstance(r["maxabs"], str) else r["maxabs"]
        print(f"{nm:<30}{fmt(r['res'])}{str(r['params']):>9}{mx:>10}")
    print("  (전체 = 이상 vs 일반+무해 음성, 일반음성 = 이상 vs 일반 정상, 무해변동 = 이상 vs 대축으로 크게 이동한 정상, "
          "d = 이상의 마할라노비스 거리 구간)")

    verdict = {}
    thr = rows["센서별 임계값(max|z|)"]["res"]["all"]
    ref_name = max(("마할라노비스 16차원", "비선형 baseline(HGB)"), key=lambda k: rows.get(k, {"res": {"all": -1}})["res"]["all"])
    ref = rows[ref_name]["res"]["all"]
    if "res_square" in nn_res:
        rs = nn_res["res_square"]
        verdict["A"] = (rs["all"] - thr >= 0.20, f"res_square {rs['all']:.3f} - 센서별 임계값 {thr:.3f} = {rs['all'] - thr:+.3f} (기준 +0.20)")
        verdict["B"] = (rs["all"] >= 0.9 * ref, f"res_square {rs['all']:.3f} / 참고 상한({ref_name}) {ref:.3f} = {rs['all'] / ref:.2f} (기준 0.90)")
        if "uniform_res_square" in nn_res:
            u = nn_res["uniform_res_square"]
            verdict["C"] = (abs(rs["all"] - u["all"]) < 0.03, f"res_square {rs['all']:.3f} vs 균등 attention {u['all']:.3f}: 차이 {rs['all'] - u['all']:+.3f} (기준 절댓값 0.03 미만이면 '평균 문맥')")
        verdict["D"] = (rs["benign"] >= 0.75, f"무해한 큰 변동 음성 상대 AUC {rs['benign']:.3f} (참고 기준 0.75)")
        print("\n=== 사전 판정 기준 적용 ===")
        for k, (ok, msg) in verdict.items():
            print(f"  {k}. {'충족' if ok else '미충족'}: {msg}")
        print("  해석: A와 B 둘 다 충족하면 '관계 이탈 탐지 능력이 확인되었다'. C 가 충족이면 '이 벤치마크에서 attention 의 이득은 센서 평균 문맥'이다.")
        print("        이 결과는 통제된 합성 벤치마크의 것이며 실제 PHM 고장 탐지를 뜻하지 않는다.")
    with open(os.path.join(args.out, "corr_break_results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), rows={k: dict(res=v["res"], params=v["params"], maxabs=v["maxabs"]) for k, v in rows.items()},
                       verdict={k: dict(ok=bool(v[0]), msg=v[1]) for k, v in verdict.items()}), f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장: {args.out}/corr_break_results.json")


if __name__ == "__main__":
    main()