#!/usr/bin/env python
"""
[실험 전용] 팀원 데이터셋(observable4_dataset_balanced.csv)에서 모델 구조별 그룹 교차검증 비교

팀원 코드(train_attention.py)를 읽고 그 프로토콜을 그대로 따르도록 맞췄다.
  - feature 열: <센서>_t-12 / _t-8 / _t-4 / _t  (센서 4개 x 시점 4개, 센서 행 우선) -> 열 이름으로 자동 탐지
  - 그룹: Event-group = pair_group, File-group = file_id
  - 분할: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cv_seed), cv_seed = 11, 23, 37, 42, 59 (5개 x 5폴드 = 25개)
  - 폴드별 표준화: 센서별 mean/std 를 학습 폴드로만 계산
  - 학습: Adam, lr 1e-3, weight decay 1e-4, batch 64, 100에폭, 마지막 에폭 모델 사용(조기 종료/검증셋 없음), 폴드 시드 = cv_seed*100 + fold
  - 보고: 25개 (시드 x 폴드) 결과의 평균 ± 표준편차 (팀원의 0.6875 ± 0.11 도 이 방식일 가능성이 큼)
  - 팀원 예측 CSV(attention_cv_predictions.csv)를 --folds-csv 로 주면 폴드를 CSV 에서 그대로 재현한다.

비교하는 모델
  logreg      16개 feature 선형 로지스틱 회귀
  hgb         HistGradientBoosting (16개 + 센서 쌍 평균 차이 6개)  <- 비선형 baseline
  attn_rtl    Q/K/V -> QK^T/2 -> softmax -> AV -> flatten -> sum(f * WFC)  (바이어스 전혀 없음)  <- 팀원 모델 재현 (RTL 구조)
  attn_fc     위와 같되 FC 에 바이어스 추가
  attn_square 위에서 FC 앞에 제곱 단계 + 바이어스
  res_fc      위에서 잔차 연결(O + X) + FC 바이어스
  res_square  잔차 + 제곱 + 바이어스  <- 우리 구조

사용법
  python exp_observable_cv.py --csv observable4_dataset_balanced.csv --out obs_cv_out
  python exp_observable_cv.py --csv ... --folds-csv attention_cv_predictions.csv      (팀원 폴드를 CSV 에서 그대로 재현)
  python exp_observable_cv.py --csv ... --models logreg,hgb                           (torch 없이 빠르게)
  python exp_observable_cv.py --csv ... --cv-seeds 11,23                              (시드 2개만: 시간 절약)

필요 패키지: numpy, pandas, scikit-learn, torch(신경망 모델에만)
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SENSORS = ["ETCHBEAMVOLTAGE", "IONGAUGEPRESSURE", "ETCHGASCHANNEL1READBACK", "FLOWCOOLPRESSURE"]
TEAM_FEATURES = [f"{s}_{t}" for s in SENSORS for t in ("t-12", "t-8", "t-4", "t")]
TEAM_CV_SEEDS = [11, 23, 37, 42, 59]

NN_KINDS = {
    "attn_rtl":    dict(res=False, head="rtl"),
    "attn_fc":     dict(res=False, head="fc"),
    "attn_square": dict(res=False, head="square"),
    "res_fc":      dict(res=True,  head="fc"),
    "res_square":  dict(res=True,  head="square"),
}
ALL_MODELS = ["logreg", "hgb", "attn_rtl", "attn_fc", "attn_square", "res_fc", "res_square"]
DEFAULT_MODELS = ["logreg", "hgb", "attn_rtl", "attn_fc", "res_square"]
THR_TEAM = 0.58
THR_HALF = 0.50
METRIC_KEYS = ("auc", "acc50", "rec50", "spec50", "acc58", "rec58", "spec58")

META_TOKENS = {"event", "index", "idx", "id", "group", "seed", "fold", "source", "name", "cat", "category", "time", "sec",
               "ttf", "label", "observability", "pair", "file", "window", "sample", "stage", "recipe", "step", "fault",
               "type", "pseudo", "number", "relative", "end", "num"}


def is_meta_col(col):
    toks = [t for t in re.split(r"[^a-z0-9]+", str(col).lower()) if t]
    return any(t in META_TOKENS for t in toks)


# ---------------------------------------------------------------------------
# 데이터 읽기
# ---------------------------------------------------------------------------
def load_csv(args):
    df = pd.read_csv(args.csv)
    for c in (args.label_col, args.file_col):
        if c not in df.columns:
            raise SystemExit(f"열 '{c}' 이 CSV 에 없습니다. 열 목록: {list(df.columns)}")
    event_col = args.event_col
    if event_col not in df.columns:
        alt = "event_group" if event_col == "pair_group" else "pair_group"
        if alt in df.columns:
            print(f"[알림] '{event_col}' 열이 없어 '{alt}' 를 사용합니다.")
            event_col = alt
        else:
            raise SystemExit(f"Event 그룹 열({args.event_col})이 없습니다. --event-col 로 지정하세요. 열 목록: {list(df.columns)}")

    if args.features:
        cols = [c.strip() for c in args.features.split(",")]
    elif all(c in df.columns for c in TEAM_FEATURES):
        cols = TEAM_FEATURES
        print("[알림] 팀원 열 이름 규칙(<센서>_t-12/t-8/t-4/t)으로 feature 16개를 찾았습니다.")
    else:
        meta = {args.label_col, event_col, args.file_col}
        num = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
        if args.feature_prefix:
            num = [c for c in num if str(c).startswith(args.feature_prefix)]
        if len(num) != 16:
            kept = [c for c in num if not is_meta_col(c)]
            if len(kept) == 16:
                print(f"[알림] 숫자 열 {len(num)}개 중 id/인덱스성 열 {[c for c in num if c not in kept]} 를 feature 에서 제외했습니다.")
                num = kept
        if len(num) != 16:
            raise SystemExit(
                f"feature 열을 자동으로 찾지 못했습니다 (후보 {len(num)}개, 16개여야 함).\n"
                f"  후보: {num}\n  전체 열: {list(df.columns)}\n"
                "  --features 열1,...,열16 (센서0 t0..t3, 센서1 t0..t3, ... 순서) 또는 --feature-prefix 로 지정하세요.")
        cols = num
    if len(cols) != 16:
        raise SystemExit(f"--features 는 16개여야 합니다 (지금 {len(cols)}개).")
    X = df[cols].to_numpy(dtype=np.float64).reshape(-1, 4, 4)
    if args.layout == "time_sensor":
        X = X.transpose(0, 2, 1).copy()
    y = df[args.label_col].to_numpy().astype(int)
    return df, X, y, df[event_col].astype(str).to_numpy(), df[args.file_col].astype(str).to_numpy(), cols, event_col


def describe(X, y, ev, fl, cols):
    print(f"샘플 {len(y):,}개 (양성 {int(y.sum())}, 음성 {int((1 - y).sum())}), feature 열: {cols[0]} ... {cols[-1]}")
    for nm, g in (("Event 그룹", ev), ("File 그룹", fl)):
        u, cnt = np.unique(g, return_counts=True)
        pos = {k: int(y[g == k].sum()) for k in u}
        single = sum(1 for k, c in zip(u, cnt) if pos[k] in (0, c))
        print(f"  {nm}: {len(u)}개, 그룹당 샘플 최소/중앙/최대 {cnt.min()}/{int(np.median(cnt))}/{cnt.max()}, 한 클래스뿐인 그룹 {single}개")
    m, s = X.mean(axis=0), X.std(axis=0)
    print("  입력 평균 (행=센서, 열=시점):")
    for i in range(4):
        print("    " + "  ".join(f"{m[i, j]:10.3f}" for j in range(4)) + "   | 표준편차 " + "  ".join(f"{s[i, j]:9.3f}" for j in range(4)))


# ---------------------------------------------------------------------------
# 분할 / 표준화 / 지표
# ---------------------------------------------------------------------------
def make_splits(X, y, groups, cv_seed, folds):
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=cv_seed)
    return list(cv.split(X, y, groups=groups))


def load_fold_splits(path, kind, seeds, df, file_col):
    """팀원 attention_cv_predictions.csv 의 (cv_type, seed, fold, sample_index) 로 평가 폴드를 재현.
    반환: {seed: [(train_idx, test_idx), ...]}, 팀원 예측 DataFrame. 해당 cv_type 이 없으면 (None, None)."""
    pr = pd.read_csv(path)
    for c in ("cv_type", "fold", "sample_index"):
        if c not in pr.columns:
            raise SystemExit(f"--folds-csv 에 '{c}' 열이 없습니다. 열 목록: {list(pr.columns)}")
    key = "file" if kind == "file" else "event"
    kinds = [k for k in pr["cv_type"].astype(str).unique() if key in k.lower()]
    if not kinds:
        print(f"[알림] 팀원 예측 CSV 의 cv_type 값 {list(pr['cv_type'].astype(str).unique())} 에 '{key}' 가 없어 이 CV 는 직접 분할합니다.")
        return None, None
    sub = pr[pr["cv_type"].astype(str) == kinds[0]]
    all_seeds = sorted(int(v) for v in sub["seed"].unique()) if "seed" in sub.columns else [None]
    use = all_seeds if not seeds else [s for s in all_seeds if s in seeds]
    if not use:
        raise SystemExit(f"요청한 시드 {seeds} 가 CSV 에 없습니다. 있는 값: {all_seeds}")
    n = len(df)
    out = {}
    for sd in use:
        ss = sub if sd is None else sub[sub["seed"] == sd]
        idx = ss["sample_index"].to_numpy().astype(int)
        base = None
        for off in (0, 1):                                    # 0 기반/1 기반 판정 (file_id 로 확인)
            j = idx - off
            if j.min() < 0 or j.max() >= n:
                continue
            if "file_id" in ss.columns and (df[file_col].astype(str).to_numpy()[j] == ss["file_id"].astype(str).to_numpy()).mean() < 0.99:
                continue
            base = off
            break
        if base is None:
            raise SystemExit("sample_index 와 CSV 행을 맞추지 못했습니다 (file_id 불일치). 인덱스 기준을 팀원에게 확인하세요.")
        j = idx - base
        if len(np.unique(j)) != len(j):
            raise SystemExit(f"seed {sd}: 한 샘플이 여러 번 나옵니다. CSV 형식을 확인하세요.")
        folds = []
        for f in sorted(ss["fold"].unique()):
            te = np.sort(j[ss["fold"].to_numpy() == f])
            folds.append((np.setdiff1d(np.arange(n), te), te))
        out[sd] = folds
    return out, sub


def standardize(Xtr, Xte):
    mu = Xtr.mean(axis=(0, 2), keepdims=True)                 # 센서(행)별
    sd = Xtr.std(axis=(0, 2), keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fold_metrics(y, p):
    if len(np.unique(y)) < 2:
        return None
    out = dict(auc=float(roc_auc_score(y, p)))
    for tag, thr in (("50", THR_HALF), ("58", THR_TEAM)):
        pred = p >= thr
        out["acc" + tag] = float((pred == y).mean())
        out["rec" + tag] = float(pred[y == 1].mean())
        out["spec" + tag] = float((~pred[y == 0]).mean())
    return out


def hgb_features(X):
    f = X.reshape(len(X), 16)
    m = X.mean(axis=2)
    diffs = [m[:, i] - m[:, j] for i in range(4) for j in range(i + 1, 4)]
    return np.column_stack([f] + diffs)


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def fit_predict_sk(name, Xtr, ytr, Xte, seed):
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr.reshape(len(Xtr), 16), ytr)
        return clf.predict_proba(Xte.reshape(len(Xte), 16))[:, 1]
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed)
    clf.fit(hgb_features(Xtr), ytr)
    return clf.predict_proba(hgb_features(Xte))[:, 1]


def fit_predict_nn(kind, Xtr, ytr, Xte, args, seed):
    """팀원 train_model 과 같은 설정으로 학습. 반환: (평가 폴드 확률, 중간값 최대 |값|)"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    spec = NN_KINDS[kind]
    seed = int(seed)                                           # numpy 정수가 들어와도 torch 가 받도록
    torch.manual_seed(seed)
    np.random.seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = nn.Parameter(torch.empty(4, 4))
            self.wk = nn.Parameter(torch.empty(4, 4))
            self.wv = nn.Parameter(torch.empty(4, 4))
            for w in (self.wq, self.wk, self.wv):
                nn.init.xavier_uniform_(w)
            if spec["head"] == "fc":
                self.fc = nn.Linear(16, 1)
            else:
                self.wfc = nn.Parameter(torch.empty(16))
                nn.init.uniform_(self.wfc, -0.1, 0.1)
                if spec["head"] == "square":
                    self.b = nn.Parameter(torch.zeros(()))

        def forward(self, x):
            q, k, v = x @ self.wq, x @ self.wk, x @ self.wv
            a = torch.softmax((q @ k.transpose(1, 2)) / 2.0, dim=-1)
            o = a @ v
            if spec["res"]:
                o = o + x
            f = o.reshape(x.shape[0], 16)
            if spec["head"] == "rtl":
                z = torch.sum(f * self.wfc, dim=1)               # 팀원 모델: 바이어스 없음
                mids = (q, k, v, o)
            elif spec["head"] == "fc":
                z = self.fc(f).squeeze(-1)
                mids = (q, k, v, o)
            else:
                sq = f ** 2
                z = sq @ self.wfc + self.b
                mids = (q, k, v, o, sq)
            return z, mids

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, generator=gen)
    net.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            z, _ = net(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(z, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        z, mids = net(torch.tensor(Xte, dtype=torch.float32))
        mx = max(float(z.abs().max()), *[float(t.abs().max()) for t in mids])
        p = torch.sigmoid(z).numpy().astype(np.float64)
    return p, mx


# ---------------------------------------------------------------------------
# 교차검증 실행 (팀원 프로토콜: 시드 x 폴드)
# ---------------------------------------------------------------------------
def run_cv(kind, cv_name, groups, X, y, models, args, df, nn_fn=fit_predict_nn):
    if args.folds_csv:
        seeds_req = [int(s) for s in args.folds_seed.split(",")] if args.folds_seed else None
        fs, sub = load_fold_splits(args.folds_csv, kind, seeds_req, df, args.file_col)
    else:
        fs, sub = None, None
    if fs is None:
        seeds = [int(s) for s in args.cv_seeds.split(",")]
        fs = {sd: make_splits(X, y, groups, sd, args.folds) for sd in seeds}
        src = "StratifiedGroupKFold(shuffle) 로 직접 분할"
    else:
        src = "팀원 예측 CSV 의 폴드를 그대로 사용"
        teammate_recompute(sub, kind)
    n_fold = sum(len(v) for v in fs.values())
    print(f"\n=== {cv_name}: 그룹 {len(np.unique(groups))}개, 시드 {[int(k) for k in fs]} x {args.folds}폴드 = {n_fold}개 ({src}) ===", flush=True)

    folds_out = {m: [] for m in models}
    maxabs = {m: 0.0 for m in models}
    secs = {m: 0.0 for m in models}
    oof_sum = {m: np.zeros(len(y)) for m in models}
    oof_cnt = np.zeros(len(y))
    for sd, splits_ in fs.items():
        for fi, (tr, te) in enumerate(splits_, start=1):
            Xtr, Xte = standardize(X[tr], X[te])
            oof_cnt[te] += 1
            for m in models:
                t0 = time.time()
                if m in NN_KINDS:
                    p, mx = nn_fn(m, Xtr, y[tr], Xte, args, int(sd) * 100 + fi)
                    maxabs[m] = max(maxabs[m], mx)
                else:
                    p = fit_predict_sk(m, Xtr, y[tr], Xte, sd)
                met = fold_metrics(y[te], p)
                if met is not None:
                    folds_out[m].append(dict(seed=sd, fold=fi, **met))
                oof_sum[m][te] += p
                secs[m] += time.time() - t0
        print(f"  시드 {sd} 완료 " + " | ".join(
            f"{m} {np.mean([r['auc'] for r in folds_out[m] if r['seed'] == sd]):.3f}" for m in models if folds_out[m]), flush=True)

    results = {}
    for m in models:
        fo = folds_out[m]
        if not fo:
            print(f"  [{m}] 평가 가능한 폴드가 없습니다.")
            continue
        agg = {k: (float(np.mean([r[k] for r in fo])), float(np.std([r[k] for r in fo]))) for k in METRIC_KEYS}
        per_seed = {int(sd): float(np.mean([r["auc"] for r in fo if r["seed"] == sd])) for sd in fs}
        results[m] = dict(agg=agg, per_seed_auc=per_seed, n_folds=len(fo), maxabs=maxabs[m], sec=secs[m])
    oof = {m: np.where(oof_cnt > 0, oof_sum[m] / np.maximum(oof_cnt, 1), np.nan) for m in models}
    return results, oof


def teammate_recompute(sub, kind):
    """팀원 예측 CSV 로 (시드, 폴드)별 지표를 다시 계산 -> 보고된 값(AUC 0.6875±0.11, Acc 0.6351±0.0978)과 비교."""
    if sub is None or not {"fold", "y_true", "probability"} <= set(sub.columns):
        return
    rows = []
    keys = ["seed", "fold"] if "seed" in sub.columns else ["fold"]
    for _, g in sub.groupby(keys):
        yt, p = g["y_true"].to_numpy().astype(int), g["probability"].to_numpy()
        m = fold_metrics(yt, p)
        if m:
            rows.append(m)
    if rows:
        s = lambda k: f"{np.mean([r[k] for r in rows]):.4f} ± {np.std([r[k] for r in rows]):.4f}"
        print(f"  [팀원 예측 CSV 재계산 | {kind}] (시드x폴드) {len(rows)}개: AUC {s('auc')}, Acc@0.5 {s('acc50')}, "
              f"Recall@0.5 {s('rec50')}, Spec@0.5 {s('spec50')}")
        print("   -> 팀원이 보고한 값과 일치하면 폴드/지표 정의를 제대로 이해한 것입니다.")


def print_summary(cv_name, res):
    print(f"\n--- {cv_name} 요약 (모든 시드x폴드의 평균 ± 표준편차) ---")
    print(f"{'모델':<13}{'AUC':>16}{'Acc@.5':>8}{'Rec@.5':>8}{'Spec@.5':>9}{'Acc@.58':>9}{'Rec@.58':>9}{'Spec@.58':>10}{'시드별 AUC 평균':>34}{'중간값최대':>9}")
    for m, r in res.items():
        a = r["agg"]
        ps = " ".join(f"{v:.2f}" for v in r["per_seed_auc"].values())
        mx = f"{r['maxabs']:.1f}" if m in NN_KINDS else "-"
        print(f"{m:<13}{a['auc'][0]:>9.3f} ± {a['auc'][1]:.3f}{a['acc50'][0]:>8.3f}{a['rec50'][0]:>8.3f}{a['spec50'][0]:>9.3f}"
              f"{a['acc58'][0]:>9.3f}{a['rec58'][0]:>9.3f}{a['spec58'][0]:>10.3f}  {ps:<32}{mx:>9}")


def breakdown(df, oof, models, cols):
    for col in cols:
        if col not in df.columns:
            continue
        print(f"\n--- '{col}' 별 AUC (시드 평균 예측, 해당 범주의 양성·음성만) ---")
        vals = df[col].astype(str).to_numpy()
        y = df["__y"].to_numpy()
        print(f"{col:<28}{'n':>6}{'양성':>6}" + "".join(f"{m:>13}" for m in models))
        for v in sorted(np.unique(vals)):
            msk = vals == v
            if msk.sum() < 20 or len(np.unique(y[msk])) < 2:
                continue
            row = f"{v[:27]:<28}{int(msk.sum()):>6}{int(y[msk].sum()):>6}"
            for m in models:
                p = oof[m][msk]
                row += f"{roc_auc_score(y[msk], p):>13.3f}" if not np.isnan(p).any() else f"{'-':>13}"
            print(row)


def time_shortcut(df, args, y, fl):
    """시간 위치만으로 양성/음성을 얼마나 맞히는가 (센서 값 미사용). 높으면 시기·단계 교란 의심."""
    cols = [c for c in (args.time_col,) if c]
    if not cols:
        return
    if args.time_col not in df.columns:
        print(f"\n[건너뜀] 시간 열 '{args.time_col}' 이 CSV 에 없습니다. 열 목록: {list(df.columns)}")
        return
    t = pd.to_numeric(df[args.time_col], errors="coerce").to_numpy()
    rank = pd.Series(t).groupby(fl).rank(pct=True).to_numpy()
    print("\n=== 시간 위치만으로 맞히기 (센서 값 미사용, 파일 그룹 CV, 시드 11) ===")
    for nm, feat in (("시간 값 자체", t), ("파일 안의 시간 순위(0~1)", rank)):
        ok = ~np.isnan(feat)
        ft, yy, g = feat[ok], y[ok], fl[ok]
        aucs = []
        for tr, te in make_splits(np.zeros(len(yy)), yy, g, 11, args.folds):
            if len(np.unique(yy[te])) < 2:
                continue
            clf = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(ft[tr].reshape(-1, 1), yy[tr])
            aucs.append(roc_auc_score(yy[te], clf.predict_proba(ft[te].reshape(-1, 1))[:, 1]))
        if aucs:
            print(f"  {nm}: AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  (0.5 에 가까워야 시간 교란이 없음)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="obs_cv_out")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS), help="쉼표 구분 또는 all. 가능: " + ",".join(ALL_MODELS))
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--event-col", default="pair_group")
    ap.add_argument("--file-col", default="file_id")
    ap.add_argument("--features", default=None, help="feature 열 16개 (쉼표 구분, 센서0 t-12..t, 센서1 ... 순서)")
    ap.add_argument("--feature-prefix", default=None)
    ap.add_argument("--layout", choices=["sensor_time", "time_sensor"], default="sensor_time")
    ap.add_argument("--time-col", default=None, help="시간 열 (예: relative_time, window_end_time): shortcut 점검용, 선택")
    ap.add_argument("--cv-seeds", default=",".join(map(str, TEAM_CV_SEEDS)), help="StratifiedGroupKFold 시드 목록 (팀원: 11,23,37,42,59)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--folds-csv", default=None, help="팀원 attention_cv_predictions.csv: 이 파일의 폴드를 그대로 재현")
    ap.add_argument("--folds-seed", default=None, help="--folds-csv 안에서 쓸 시드 (쉼표 구분, 기본: 전부)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--by", default="fault_name,observability_category", help="범주별 AUC 를 볼 열 (없는 열은 건너뜀)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    models = ALL_MODELS if args.models.strip() == "all" else [m.strip() for m in args.models.split(",")]
    for m in models:
        if m not in ALL_MODELS:
            raise SystemExit(f"알 수 없는 모델 {m}. 가능: {', '.join(ALL_MODELS)}")

    df, X, y, ev, fl, cols, event_col = load_csv(args)
    describe(X, y, ev, fl, cols)
    df = df.assign(__y=y)
    all_res = {}
    for kind, cv_name, g in (("file", f"File-group CV ({args.file_col})", fl), ("event", f"Event-group CV ({event_col})", ev)):
        res, oof = run_cv(kind, cv_name, g, X, y, models, args, df)
        print_summary(cv_name, res)
        pd.DataFrame({"row": np.arange(len(y)), "group": g, "label": y, **{f"p_{m}": oof[m] for m in models}}) \
            .to_csv(os.path.join(args.out, f"oof_{kind}.csv"), index=False)
        if kind == "file":
            breakdown(df, oof, [m for m in models if m in res], [c.strip() for c in args.by.split(",") if c.strip()])
        all_res[cv_name] = res
    time_shortcut(df, args, y, fl)
    with open(os.path.join(args.out, "obs_cv_results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), results=all_res), f, ensure_ascii=False, indent=2)
    print(f"\n저장: {args.out}/obs_cv_results.json, oof_file.csv, oof_event.csv (시드 평균 예측 확률)")
    print("읽는 법: 표준편차는 시드x폴드 전체의 값이라 팀원이 보고한 방식과 같다. 모델 간 AUC 차이가 표준편차보다 작으면 우열을 단정하지 말 것.")


if __name__ == "__main__":
    main()