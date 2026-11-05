#!/usr/bin/env python
"""
[실험] 최종 모델을 어떤 데이터셋으로 학습할지 교차 평가로 정한다.

학습 구성 3가지
  corr : dataset_corr 로 학습   (센서 쌍의 정상 관계를 깨는 이상 + 어려운 음성)
  old  : dataset_t 로 학습      (기존 합성: offset / ramp / swap_cond)
  mix  : 위 두 학습 세트를 각각 50% 씩 섞어 학습
평가 세트 2가지 (모든 구성을 같은 두 세트에서 평가)
  corr 평가: dataset_corr 의 Test  (전체, 무해한 큰 변동 음성만 상대로 한 AUC)
  old  평가: dataset_t_hard 의 Test (전체, offset / ramp / swap_cond 유형별 AUC)
  (두 데이터셋은 같은 정상 윈도우와 같은 표준화 통계를 쓰므로 한 모델에 그대로 넣을 수 있다.)

선택 규칙 (결과를 보기 전에 정해 둔 것)
  1순위: 두 평가 세트의 '전체 AUC' 중 낮은 값이 가장 높은 구성 (한쪽에만 강한 모델을 피한다)
  제외 조건: 중간값 최대 |값| 이 100 을 넘는 구성 (Q8.8 범위 위험)
  참고(선택에 쓰지 않음): 슬라이더 시연과 가까운 'old 평가의 offset AUC', corr 평가의 무해한 큰 변동 음성 상대 AUC

사용법
  python exp_cross_train.py --corr dataset_corr --old dataset_t --old-test dataset_t_hard --out cross_out
  python exp_cross_train.py ... --models res_square,rtl      (구조 2개)

필요 패키지: numpy, scikit-learn, torch. exp_head.py, make_dataset_corr.py 를 같은 폴더에 둔다.
"""
import argparse
import json
import os
import time

import numpy as np

import exp_head as eh
import make_dataset_corr as mdc

OLD_TYPES = ["offset", "ramp", "swap_cond"]
MAXABS_LIMIT = 100.0


def load_split(d, split):
    z = np.load(os.path.join(d, f"dataset_{split}.npz"))
    return {k: z[k] for k in z.files}


def subsample(d, frac, rng):
    idx = rng.permutation(len(d["y"]))[: int(len(d["y"]) * frac)]
    return {"X": d["X"][idx], "y": d["y"][idx]}


def cat(a, b):
    return {"X": np.concatenate([a["X"], b["X"]]), "y": np.concatenate([a["y"], b["y"]])}


def old_report(te, z):
    y, at = te["y"], te["atype"]
    out = {"all": mdc.auc_safe(y, z)}
    for t, nm in enumerate(OLD_TYPES):
        m = (y == 0) | (at == t)
        out[nm] = mdc.auc_safe(y[m], z[m]) if (at == t).sum() > 20 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", default="dataset_corr")
    ap.add_argument("--old", default="dataset_t", help="기존 합성 학습/검증 세트")
    ap.add_argument("--old-test", default="dataset_t_hard", help="기존 합성 평가 세트")
    ap.add_argument("--out", default="cross_out")
    ap.add_argument("--models", default="res_square")
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
    rng = np.random.default_rng(args.seed)

    c_tr, c_va, c_te = load_split(args.corr, "train"), load_split(args.corr, "val"), load_split(args.corr, "test")
    o_tr, o_va = load_split(args.old, "train"), load_split(args.old, "val")
    o_te = load_split(args.old_test, "test")
    sets = {
        "corr": (c_tr, c_va),
        "old": (o_tr, o_va),
        "mix": (cat(subsample(c_tr, 0.5, rng), subsample(o_tr, 0.5, rng)), cat(subsample(c_va, 0.5, rng), subsample(o_va, 0.5, rng))),
    }
    print(f"학습 세트 크기: corr {len(c_tr['y']):,}, old {len(o_tr['y']):,}, mix {len(sets['mix'][0]['y']):,} / "
          f"평가: corr {len(c_te['y']):,}, old {len(o_te['y']):,}")

    names = [m.strip() for m in args.models.split(",")]
    rows = []
    for mname in names:
        if mname not in eh.SPECS:
            raise SystemExit(f"알 수 없는 구조 {mname}. 가능: {', '.join(eh.SPECS)}")
        spec = eh.SPECS[mname]
        for tname, (tr, va) in sets.items():
            t0, best = time.time(), (-1.0, None)
            for r in range(max(1, args.restarts)):
                model, v_auc, ep = eh.train_one(spec, tr, va, args, args.seed + r)
                if v_auc > best[0]:
                    best = (v_auc, model)
            zc, mx1 = eh.predict(best[1], c_te["X"])
            zo, mx2 = eh.predict(best[1], o_te["X"])
            rc, ro = mdc.report_scores(c_te, zc), old_report(o_te, zo)
            rows.append(dict(model=mname, train=tname, corr_all=rc["all"], corr_benign=rc["benign"], old_all=ro["all"],
                             old_offset=ro["offset"], old_ramp=ro["ramp"], old_swap=ro["swap_cond"], maxabs=max(mx1, mx2),
                             val_auc=best[0], sec=time.time() - t0))
            print(f"[{mname} / {tname}] corr {rc['all']:.3f}, old {ro['all']:.3f}  ({time.time() - t0:.0f}초)", flush=True)

    print("\n=== 교차 평가 (Test AUC) ===")
    print(f"{'구조':<12}{'학습':<6}{'corr 전체':>10}{'corr 무해':>10}{'old 전체':>10}{'old offset':>11}{'old ramp':>10}{'old swap':>10}{'두 세트 중 낮은 값':>20}{'중간값최대':>10}")
    for r in rows:
        lo = min(r["corr_all"], r["old_all"])
        r["worst"] = lo
        print(f"{r['model']:<12}{r['train']:<6}{r['corr_all']:>10.3f}{r['corr_benign']:>10.3f}{r['old_all']:>10.3f}{r['old_offset']:>11.3f}"
              f"{r['old_ramp']:>10.3f}{r['old_swap']:>10.3f}{lo:>20.3f}{r['maxabs']:>10.1f}")
    ok = [r for r in rows if r["maxabs"] <= MAXABS_LIMIT]
    print("\n=== 선택 규칙 적용 ===")
    if not ok:
        print("  중간값 최대가 100 이하인 구성이 없습니다. --hw-penalty 를 높이거나 학습 설정을 점검하세요.")
        choice = None
    else:
        choice = max(ok, key=lambda r: r["worst"])
        for r in sorted(ok, key=lambda r: -r["worst"]):
            print(f"  {r['model']}/{r['train']}: 두 세트 중 낮은 AUC {r['worst']:.3f}")
        print(f"  -> 선택: 구조 {choice['model']}, 학습 데이터 {choice['train']}  "
              f"(참고: old offset {choice['old_offset']:.3f}, corr 무해 음성 상대 {choice['corr_benign']:.3f})")
        print("  해석: 선택된 구성이 corr 평가와 old 평가 양쪽에서 가장 안정적이라는 뜻이다. 한쪽 데이터셋에만 강한 구성은 선택되지 않는다.")
    with open(os.path.join(args.out, "cross_results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), rows=rows, choice=choice), f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장: {args.out}/cross_results.json")


if __name__ == "__main__":
    main()