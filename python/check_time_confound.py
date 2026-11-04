#!/usr/bin/env python
"""
[진단] 시간 위치(relative_time) 단서 점검

배경: relative_time 하나만으로 양성/음성을 가르면 AUC 0.75 가 나왔다 (센서 값 미사용).
      -> 양성(실제 고장 직전 윈도우)과 음성(matched normal)의 '윈도우 위치 분포'가 다르다는 뜻.
이 스크립트는 (1) 라벨별 위치 분포, (2) event 당 윈도우 수, (3) 모델이 그 위치 단서를 쓰는지(위치 구간 안에서의 AUC)를 본다.

사용법
  python check_time_confound.py --csv observable4_dataset_balanced.csv --oof obs_cv_out/oof_file.csv
  (oof 파일은 exp_observable_cv.py 가 만든 시드 평균 예측. 없으면 (1),(2)만 출력)

읽는 법
  - (1)에서 양성과 음성의 분위수/구간별 개수가 크게 다르면 데이터 구성 때문에 생긴 위치 차이.
  - (3)에서 '구간 안 AUC'가 '전체 AUC'보다 눈에 띄게 낮으면 모델이 위치(단계 위상) 단서를 일부 쓰고 있다는 뜻.
    비슷하면 모델은 위치와 무관하게 센서 값으로 판단하고 있다.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--oof", default=None)
    ap.add_argument("--time-col", default="relative_time")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--event-col", default="pair_group")
    ap.add_argument("--bins", type=int, default=6)
    ap.add_argument("--min-bin", type=int, default=30, help="구간 안 AUC 를 계산할 최소 샘플 수")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    y = df[a.label_col].to_numpy().astype(int)
    t = pd.to_numeric(df[a.time_col], errors="coerce")
    print(f"샘플 {len(df):,}개, 시간 열 '{a.time_col}' 범위 {t.min():.1f} ~ {t.max():.1f}, 고유값 {t.nunique()}개")

    print("\n(1) 라벨별 위치 분포 (분위수)")
    qs = [0, .1, .25, .5, .75, .9, 1]
    tab = pd.DataFrame({("양성" if lab == 1 else "음성"): t[y == lab].quantile(qs).to_numpy() for lab in (1, 0)},
                       index=[f"{int(q * 100)}%" for q in qs])
    print(tab.round(1).to_string())
    edges = np.linspace(t.min(), t.max(), a.bins + 1)
    cut = pd.cut(t, edges, include_lowest=True)
    ct = pd.crosstab(cut, df[a.label_col])
    ct.columns = ["음성" if c == 0 else "양성" for c in ct.columns]
    ct["양성 비율"] = (ct["양성"] / (ct["양성"] + ct["음성"])).round(3)
    print("\n    위치 구간별 개수 (양성 비율이 0.5 에서 멀수록 위치가 라벨을 알려 줌)")
    print(ct.to_string())

    print("\n(2) event 당 윈도우 수 (pair_group x 라벨)")
    cnt = df.groupby([a.event_col, a.label_col]).size().unstack(fill_value=0)
    cnt.columns = ["음성" if c == 0 else "양성" for c in cnt.columns]
    print(cnt.describe().loc[["min", "25%", "50%", "75%", "max"]].round(1).to_string())

    if a.oof:
        o = pd.read_csv(a.oof)
        if len(o) != len(df):
            raise SystemExit(f"oof 행 수({len(o)}) 와 CSV 행 수({len(df)}) 가 다릅니다.")
        print("\n(3) 모델의 위치 의존성: 전체 AUC vs 위치 구간 안에서의 AUC (표본 가중 평균)")
        print(f"{'모델':<14}{'전체 AUC':>10}{'구간 안 AUC':>13}{'차이':>8}")
        for c in [c for c in o.columns if c.startswith("p_")]:
            p = o[c].to_numpy()
            ok = ~np.isnan(p)
            overall = roc_auc_score(y[ok], p[ok])
            num = den = 0.0
            for _, idx in pd.Series(np.arange(len(df))).groupby(cut.to_numpy()):
                ii = idx.to_numpy()
                ii = ii[ok[ii]]
                if len(ii) >= a.min_bin and len(np.unique(y[ii])) == 2:
                    num += roc_auc_score(y[ii], p[ii]) * len(ii)
                    den += len(ii)
            within = num / den if den else float("nan")
            print(f"{c[2:]:<14}{overall:>10.3f}{within:>13.3f}{within - overall:>8.3f}")


if __name__ == "__main__":
    main()