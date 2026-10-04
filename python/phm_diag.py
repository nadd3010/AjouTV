#!/usr/bin/env python
"""
PHM 2018 데이터 진단 스크립트 v2 (교수님 피드백 3, 4번 대응)

목적: 고장까지 남은 시간(TTF) 구간별로, 정상 구간과 실제로 구분되는지 확인한다.
      (Attention 모델 없이 로지스틱 회귀 baseline만 사용)
v2 변경점
  - shutter 값이 0/1이 아닌 raw 값(0,1,2,3,255)이어도 멈추지 않음. 값 분포를 출력하고
    --shutter-open 으로 '식각 중' 값을 지정 (기본 1)
  - stage / recipe / recipe_step 을 캐시에 저장하고, 비정상 윈도우와 '같은 공정 조건'의
    정상 윈도우만 비교하는 매칭 모드 추가

사용법
  1) 캐시 만들기 (CSV를 한 번 읽어서 4스텝 윈도우를 npz로 저장; 오래 걸림)
     python phm_diag.py build --base D:\\...\\phm_data_challenge_2018\\train --cache cache --max-files 4
  2) 분석 (캐시만 읽으므로 빠름, 파라미터를 바꿔 여러 번 실행 가능)
     python phm_diag.py analyze --cache cache

필요 패키지: pip install pandas numpy scikit-learn matplotlib
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

SENSOR_ALL = ["IONGAUGEPRESSURE", "ETCHBEAMVOLTAGE", "ETCHGASCHANNEL1READBACK",
              "FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE", "ETCHSUPPRESSORCURRENT"]
SETS = {
    "current4": ["ETCHBEAMVOLTAGE", "IONGAUGEPRESSURE",
                 "ETCHGASCHANNEL1READBACK", "FLOWCOOLPRESSURE"],
    "alt4": ["FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE",
             "IONGAUGEPRESSURE", "ETCHSUPPRESSORCURRENT"],
}
COND_COLS = ["stage", "recipe", "recipe_step"]
FAULT_LABELS = ["pressure_low", "pressure_high", "leak"]
BIN_EDGES = [0, 300, 600, 1200, 2400, 4800, 9600]   # TTF 구간 경계(초)
MODES = ["none", "shutter", "match", "shutter+match"]
STEP = 4        # 샘플링 간격(초)
WIN = 4         # 윈도우 길이(스텝)


def fault_type(name):
    s = str(name).lower()
    if "dropped" in s:
        return 0
    if "too high" in s:
        return 1
    if "leak" in s:
        return 2
    return -1


def next_gap(ftimes, te):
    """te 이후 가장 가까운 fault까지의 시간(초). 없으면 inf."""
    out = np.full(len(te), np.inf)
    if len(ftimes) == 0:
        return out
    idx = np.searchsorted(ftimes, te, side="left")
    m = idx < len(ftimes)
    out[m] = ftimes[idx[m]] - te[m]
    return out


def prev_gap(ftimes, te):
    """te 이전 가장 가까운 fault 이후 경과 시간(초). 없으면 inf."""
    out = np.full(len(te), np.inf)
    if len(ftimes) == 0:
        return out
    idx = np.searchsorted(ftimes, te, side="right") - 1
    m = idx >= 0
    out[m] = te[m] - ftimes[idx[m]]
    return out


# ----------------------------------------------------------------------------
# 1단계: build
# ----------------------------------------------------------------------------
def build(args):
    os.makedirs(args.cache, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.base, "*_DC_train.csv")))
    if not files:
        sys.exit("센서 CSV를 찾지 못했습니다. --base 경로를 확인하세요: " + args.base)
    if args.max_files:
        files = files[:args.max_files]
    rng = np.random.default_rng(0)
    cols = ["time", "FIXTURESHUTTERPOSITION"] + COND_COLS + SENSOR_ALL
    dtypes = {c: "float32" for c in cols}
    dtypes["time"] = "int64"

    for f in files:
        name = os.path.basename(f).replace("_DC_train.csv", "")
        out = os.path.join(args.cache, name + ".npz")
        if os.path.exists(out):
            print(f"[skip] {name} (캐시 있음)")
            continue
        fpath = os.path.join(args.base, "train_faults", name + "_train_fault_data.csv")
        if not os.path.exists(fpath):
            print(f"[경고] 고장 파일 없음, 건너뜀: {fpath}")
            continue

        t0 = time.time()
        print(f"[read] {name} ...", flush=True)
        df = pd.read_csv(f, usecols=cols, dtype=dtypes)
        n = len(df)
        t = df["time"].to_numpy()
        X = df[SENSOR_ALL].to_numpy(np.float32)
        sh = df["FIXTURESHUTTERPOSITION"].to_numpy(np.float32)
        cond = df[COND_COLS].to_numpy(np.float32)
        del df

        vc = pd.Series(sh).value_counts(dropna=False).head(6)
        print(f"  행 수={n:,}")
        print("  shutter 값 분포: " + ", ".join(f"{k}={v / n:.1%}" for k, v in vc.items()))
        print(f"  '식각 중' shutter 값을 {args.shutter_open} 로 사용 (바꾸려면 --shutter-open)")
        print(f"  stage 종류 {len(np.unique(cond[:, 0][~np.isnan(cond[:, 0])]))}개, "
              f"recipe 종류 {len(np.unique(cond[:, 1][~np.isnan(cond[:, 1])]))}개, "
              f"recipe_step 종류 {len(np.unique(cond[:, 2][~np.isnan(cond[:, 2])]))}개")

        dt = np.diff(t)
        print(f"  dt==4초 비율={np.mean(dt == STEP):.3f}, 큰 공백(>60초) 개수={int(np.sum(dt > 60))}")

        fl = pd.read_csv(fpath)
        ftypes = fl["fault_name"].map(fault_type).to_numpy()
        ftimes = fl["time"].to_numpy().astype(np.int64)
        order = np.argsort(ftimes)
        ftimes, ftypes = ftimes[order], ftypes[order]
        cnt = {FAULT_LABELS[k]: int(np.sum(ftypes == k)) for k in range(3)}
        print(f"  고장 개수: {cnt} (분류 불가 {int(np.sum(ftypes == -1))})")
        if len(ftimes) > 1:
            g = np.diff(ftimes)
            print(f"  직전 고장과의 간격 ≤2400초: {np.mean(g <= 2400):.2%}, "
                  f"≤9600초: {np.mean(g <= 9600):.2%}  (고장 {len(ftimes)}개 중)")

        # 비중첩 4스텝 연속 윈도우 (시간 연속성 검사)
        ok = (t[WIN - 1:] - t[:n - WIN + 1]) == STEP * (WIN - 1)
        cand = np.flatnonzero(ok)
        cand = cand[cand % WIN == 0]
        t_end = t[cand + WIN - 1]

        ttf_any = next_gap(ftimes, t_end)
        ttf_k = np.stack([next_gap(ftimes[ftypes == k], t_end) for k in range(3)], axis=1)
        since = prev_gap(ftimes, t_end)

        zone = ttf_any <= args.max_ttf
        normal = (ttf_any >= args.normal_min) & (since >= args.since_min)
        idx_n = np.flatnonzero(normal)
        if len(idx_n) > args.normal_cap:
            idx_n = rng.choice(idx_n, args.normal_cap, replace=False)
        keep = np.zeros(len(cand), bool)
        keep[np.flatnonzero(zone)] = True
        keep[idx_n] = True
        sel = np.flatnonzero(keep)
        print(f"  윈도우: 전체 {len(cand):,} → 보관 {len(sel):,} "
              f"(고장 전 {int(zone.sum()):,}, 정상풀 {len(idx_n):,})")

        ix = cand[sel][:, None] + np.arange(WIN)[None, :]
        Xw = X[ix]                                        # (m, 4, K)
        shw = (sh[ix] == args.shutter_open).mean(axis=1)  # 열림 비율
        cw = cond[ix]                                     # (m, 4, 3)
        c_last = cw[:, -1, :]
        c_const = (np.nan_to_num(cw, nan=-9) == np.nan_to_num(cw[:, :1, :], nan=-9)).all(axis=(1, 2)) \
            & ~np.isnan(cw).any(axis=(1, 2))
        np.savez_compressed(
            out, t_end=t_end[sel], ttf_any=ttf_any[sel], ttf_k=ttf_k[sel],
            since=since[sel], shut_frac=shw.astype(np.float32), Xw=Xw.astype(np.float32),
            cond=c_last.astype(np.float32), cond_const=c_const,
            sensors=np.array(SENSOR_ALL))
        print(f"  저장: {out}  ({time.time() - t0:.0f}초)\n")
    print("build 완료. 이어서: python phm_diag.py analyze --cache", args.cache)


# ----------------------------------------------------------------------------
# 2단계: analyze
# ----------------------------------------------------------------------------
def load_cache(cache):
    fs = sorted(glob.glob(os.path.join(cache, "*.npz")))
    if not fs:
        sys.exit("캐시가 없습니다. 먼저 build를 실행하세요.")
    keys = ["t_end", "ttf_any", "ttf_k", "since", "shut_frac", "Xw", "cond", "cond_const"]
    parts = {k: [] for k in keys + ["file"]}
    names = []
    for i, f in enumerate(fs):
        z = np.load(f)
        if "cond" not in z.files:
            sys.exit(f"{os.path.basename(f)} 는 v1 캐시입니다. cache 폴더를 지우고 build를 다시 실행하세요.")
        names.append(os.path.basename(f)[:-4])
        for k in keys:
            parts[k].append(z[k])
        parts["file"].append(np.full(len(z["t_end"]), i))
    D = {k: np.concatenate(v) for k, v in parts.items()}
    c = D["cond"]
    ok = D["cond_const"] & ~np.isnan(c).any(axis=1)
    cc = np.nan_to_num(c, nan=0).astype(np.int64)
    D["key"] = np.where(ok, (cc[:, 0] * 1000 + cc[:, 1]) * 1000 + cc[:, 2], -1)
    return D, names


def key_str(k):
    return f"s{k // 1_000_000}/r{(k // 1000) % 1000}/p{k % 1000}"


def feats(Xw, cols_idx):
    Xs = Xw[:, :, cols_idx]
    return np.concatenate([Xs.mean(axis=1), Xs[:, -1, :] - Xs[:, 0, :]], axis=1)


def fit_auc(Ftr, ytr, Fte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if min(ytr.sum(), (1 - ytr).sum()) < 20 or min(yte.sum(), (1 - yte).sum()) < 5:
        return np.nan
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=500, class_weight="balanced"))
    m.fit(Ftr, ytr)
    return roc_auc_score(yte, m.predict_proba(Fte)[:, 1])


def evaluate(F, y, fidx, tend, split_t, gap):
    # (a) 같은 장비 안에서 앞 70%는 학습, 뒤 30%는 평가 (사이에 gap)
    tr = tend < (split_t[fidx] - gap)
    te = tend >= split_t[fidx]
    a = fit_auc(F[tr], y[tr], F[te], y[te])
    # (b) 처음 보는 장비로 평가 (장비 단위 분리)
    files = np.unique(fidx)
    aucs = []
    if len(files) >= 2:
        for f in files:
            te = fidx == f
            v = fit_auc(F[~te], y[~te], F[te], y[te])
            if not np.isnan(v):
                aucs.append(v)
    b = float(np.mean(aucs)) if aucs else np.nan
    return a, b


def select(D, mode, ab, nm, rng, ratio=3):
    """모드에 따라 비정상/정상 윈도우를 고른다. 반환: (인덱스, 라벨)"""
    if "shutter" in mode:
        sh_ok = D["shut_frac"] == 1
        ab, nm = ab & sh_ok, nm & sh_ok
    ia, inn = np.flatnonzero(ab), np.flatnonzero(nm)
    if "match" in mode:
        key = D["key"]
        ka, kn = key[ia], key[inn]
        order = np.argsort(kn, kind="stable")
        kn_s = kn[order]
        keep_a, keep_n = [], []
        for u in np.unique(ka[ka >= 0]):
            a_idx = ia[ka == u]
            lo, hi = np.searchsorted(kn_s, u, "left"), np.searchsorted(kn_s, u, "right")
            pool = inn[order[lo:hi]]
            if len(pool) == 0:
                continue
            n_a = min(len(a_idx), len(pool))
            a_use = rng.choice(a_idx, n_a, replace=False) if n_a < len(a_idx) else a_idx
            n_use = min(len(pool), ratio * n_a)
            n_sel = rng.choice(pool, n_use, replace=False) if n_use < len(pool) else pool
            keep_a.append(a_use)
            keep_n.append(n_sel)
        if not keep_a:
            return np.array([], int), np.array([])
        ia, inn = np.concatenate(keep_a), np.concatenate(keep_n)
    else:
        cap = max(20 * len(ia), 5000)
        if len(inn) > cap:
            inn = rng.choice(inn, cap, replace=False)
    sel = np.r_[ia, inn]
    y = np.r_[np.ones(len(ia)), np.zeros(len(inn))]
    return sel, y


def analyze(args):
    from sklearn.metrics import roc_auc_score
    D, names = load_cache(args.cache)
    rng = np.random.default_rng(1)
    fidx, tend = D["file"], D["t_end"]
    print(f"장비 파일 {len(names)}개: {names}")
    print(f"윈도우 {len(tend):,}개\n")

    split_t = np.array([np.quantile(tend[fidx == i], 0.7) for i in range(len(names))])
    normal = (D["ttf_any"] >= args.normal_min) & (D["since"] >= args.since_min)
    if args.normal_local > 0:
        # 정상풀을 '고장 주변 시기'로 제한: 가장 가까운 고장(앞/뒤)과의 거리가 normal_local 이하
        # -> 장비 노후화/드리프트로 인한 시기 차이를 줄이기 위한 통제
        normal &= np.minimum(D["ttf_any"], D["since"]) <= args.normal_local
    print(f"정상풀 윈도우 {int(normal.sum()):,}개 (normal_min={args.normal_min:.0f}, "
          f"since_min={args.since_min:.0f}, normal_local={args.normal_local:.0f})")
    edges = BIN_EDGES
    bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    blabel = [f"{lo}-{hi}s" for lo, hi in bins]
    types = [("any", D["ttf_any"])] + [(FAULT_LABELS[k], D["ttf_k"][:, k]) for k in range(3)]

    # --- (1) 공정 상태 구성 비교 ---
    print("=== [1] 그룹별 공정 상태 구성 (상태가 달라지면 센서 평균이 달라 보일 수 있음) ===")
    print("shut=4스텝 모두 '식각 중' shutter 값인 비율, const=4스텝 동안 stage/recipe/step 불변 비율")
    groups = [("normal", normal)] + [(f"TTF {lb}", (D["ttf_any"] >= lo) & (D["ttf_any"] < hi))
                                     for (lo, hi), lb in zip(bins, blabel)]
    rows = []
    for gname, m in groups:
        if not m.any():
            rows.append((gname, 0, np.nan, np.nan, ""))
            continue
        ks = D["key"][m]
        ks = ks[ks >= 0]
        top = ""
        if len(ks):
            u, c = np.unique(ks, return_counts=True)
            o = np.argsort(-c)[:3]
            top = "  ".join(f"{key_str(u[i])}:{c[i] / m.sum():.0%}" for i in o)
        rows.append((gname, int(m.sum()), float((D["shut_frac"][m] == 1).mean()),
                     float(D["cond_const"][m].mean()), top))
    print(pd.DataFrame(rows, columns=["group", "n", "shut", "const", "top3 (stage/recipe/step)"])
          .to_string(index=False))

    # --- (2) 센서별 단변량 AUC ---
    print("\n=== [2] 센서별 단변량 AUC (0.5=구분 불가, 방향 무관). none vs match 비교 ===")
    for mode in ("none", "match"):
        uni = {}
        for si, s in enumerate(SENSOR_ALL):
            v = D["Xw"][:, :, si].mean(axis=1)
            finite = ~np.isnan(v)
            col = []
            for (lo, hi) in bins:
                ab = finite & (D["ttf_any"] >= lo) & (D["ttf_any"] < hi)
                nm = finite & normal
                sel, y = select(D, mode, ab, nm, rng)
                if len(y) == 0 or y.sum() < 20 or (1 - y).sum() < 20:
                    col.append(np.nan)
                    continue
                a = roc_auc_score(y, v[sel])
                col.append(max(a, 1 - a))
            uni[s] = col
        print(f"--- mode={mode} ---")
        print(pd.DataFrame(uni, index=blabel).T.round(3).to_string())

    # --- (3) 로지스틱 회귀 baseline ---
    print("\n=== [3] 로지스틱 회귀 AUC (TTF 구간별 비정상 vs 정상풀) ===")
    print("각 칸: time_split / tool_out")
    print("  time_split : 같은 장비의 앞 70%로 학습, 뒤 30%로 평가")
    print("  tool_out   : 학습에 없는 장비로 평가 (장비 평균)")
    print("mode: none=통제 없음, shutter=식각 중만, match=같은 stage/recipe/step끼리 비교, "
          "shutter+match=둘 다\n")
    res = []
    for sname, scols in SETS.items():
        cidx = [SENSOR_ALL.index(c) for c in scols]
        F_all = feats(D["Xw"], cidx)
        finite = np.isfinite(F_all).all(axis=1)
        for mode in MODES:
            for tname, ttf in types:
                for (lo, hi), lb in zip(bins, blabel):
                    ab = finite & (ttf >= lo) & (ttf < hi)
                    nm = finite & normal
                    sel, y = select(D, mode, ab, nm, rng)
                    n_ab = int(y.sum()) if len(y) else 0
                    n_nm = int((1 - y).sum()) if len(y) else 0
                    if n_ab < 30 or n_nm < 30:
                        a = b = np.nan
                    else:
                        a, b = evaluate(F_all[sel], y, fidx[sel], tend[sel], split_t, args.gap)
                    res.append(dict(set=sname, mode=mode, type=tname, bin=lb,
                                    n_abn=n_ab, n_norm=n_nm, time_split=a, tool_out=b))
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(args.cache, "auc_results.csv"), index=False)

    def fmt(x):
        return "  -  " if np.isnan(x) else f"{x:.2f}"
    for sname in SETS:
        for mode in MODES:
            print(f"--- 센서셋={sname}, mode={mode} ---")
            sub = R[(R["set"] == sname) & (R["mode"] == mode)]
            tab = {}
            for lb in blabel:
                s2 = sub[sub["bin"] == lb].set_index("type")
                tab[lb] = [f"{fmt(s2.loc[t, 'time_split'])}/{fmt(s2.loc[t, 'tool_out'])}"
                           if t in s2.index else "-" for t, _ in types]
            print(pd.DataFrame(tab, index=[t for t, _ in types]).to_string())
            nn = sub[sub["type"] == "any"].set_index("bin").loc[blabel]
            print("  (any) n_abn:", list(nn["n_abn"]), " n_norm:", list(nn["n_norm"]), "\n")

    # --- 그래프 ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
        colors = {"current4": "tab:red", "alt4": "tab:blue"}
        for ax, (tname, _) in zip(axs, types):
            for sname in SETS:
                sub = R[(R["set"] == sname) & (R["mode"] == "match") & (R["type"] == tname)]
                sub = sub.set_index("bin").loc[blabel]
                ax.plot(blabel, sub["time_split"], "-o", color=colors[sname],
                        label=f"{sname} time_split")
                ax.plot(blabel, sub["tool_out"], "--s", color=colors[sname],
                        label=f"{sname} tool_out")
            ax.axhline(0.5, color="gray", lw=0.8)
            ax.set_title(f"fault type: {tname} (matched conditions)")
            ax.set_xlabel("time to fault")
            ax.tick_params(axis="x", rotation=45)
        axs[0].set_ylabel("AUC")
        axs[0].legend(fontsize=7)
        plt.tight_layout()
        p = os.path.join(args.cache, "auc_by_ttf.png")
        plt.savefig(p, dpi=130)
        print("그래프 저장:", p)
    except Exception as e:  # noqa
        print("그래프 생성 실패(무시 가능):", e)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--base", required=True, help="train 폴더 경로")
    b.add_argument("--cache", default="cache")
    b.add_argument("--max-files", type=int, default=4, help="0이면 전체 (오래 걸림)")
    b.add_argument("--max-ttf", type=float, default=9600)
    b.add_argument("--shutter-open", type=float, default=1.0,
                   help="'식각 중'에 해당하는 FIXTURESHUTTERPOSITION 값 (기본 1)")
    b.add_argument("--normal-min", type=float, default=86400,
                   help="정상풀: 어떤 고장이든 다음 고장까지 최소 시간(초)")
    b.add_argument("--since-min", type=float, default=21600,
                   help="정상풀: 직전 고장 이후 최소 경과 시간(초)")
    b.add_argument("--normal-cap", type=int, default=200000, help="파일당 정상 윈도우 최대 개수")
    a = sp.add_parser("analyze")
    a.add_argument("--cache", default="cache")
    a.add_argument("--normal-min", type=float, default=86400)
    a.add_argument("--since-min", type=float, default=21600)
    a.add_argument("--gap", type=float, default=9600, help="time_split에서 학습/평가 사이 간격(초)")
    a.add_argument("--normal-local", type=float, default=0,
                   help="0이면 끔. 예: 259200 이면 가장 가까운 고장과 3일 이내인 정상만 사용")
    args = ap.parse_args()
    (build if args.cmd == "build" else analyze)(args)


if __name__ == "__main__":
    main()