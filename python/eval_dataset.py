#!/usr/bin/env python
"""
합성 이상 데이터셋 신뢰도 검증 (Attention 모델과 무관한 비선형 baseline 사용)

확인하는 것
  [A] 전체 성능 + 이상 종류별 + 변화 크기(eff)별 + 원인 센서별
  [B] 종류 간 일반화: 한 종류로 학습 -> 다른 종류로 평가 (주입 흔적을 외웠는지 검사)
  [C] leave-one-type-out: 두 종류로 학습 -> 안 본 종류로 평가
  [D] 변화 크기별 탐지 곡선 (그림 저장)
  [E] 실제 정상 윈도우의 오탐률 (Val에서 임계값 결정, 처음 보는 Test 장비별 FPR)
  [F] shortcut 점검: 이상 윈도우에만 센서 관측 경계값이 나타나는지
  [G] 원인 센서 복원 정확도 (비선형 baseline, 참고용 상한선)
  [H] (선택, --cache) 실제 고장 직전 윈도우 부록: 모델 점수가 정상과 구분되는지 (결과를 그대로 보고)

사용법
  python eval_dataset.py --data dataset
  python eval_dataset.py --data dataset --cache cache      # 부록 H 포함

필요 패키지: numpy, scikit-learn, matplotlib
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ATYPES = ["offset", "ramp", "swap_cond"]
EFF_BINS = [0.05, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 99]   # 낮은 구간은 --min-eff 를 낮춰 만든 데이터에서만 채워짐


def feat(X):
    n = len(X)
    pair = [X[:, i, :].mean(axis=1) - X[:, j, :].mean(axis=1)
            for i in range(4) for j in range(i + 1, 4)]
    return np.c_[X.reshape(n, -1), np.stack(pair, axis=1)]


def fit(X, y, seed=0):
    m = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
    m.fit(feat(X), y)
    return m


def score(m, X):
    return m.predict_proba(feat(X))[:, 1]


def auc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def f2(x):
    return " n/a " if np.isnan(x) else f"{x:.3f}"


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--cache", default=None, help="지정하면 부록 H(실제 고장 직전 윈도우) 실행")
    args = ap.parse_args()

    tr = load(os.path.join(args.data, "dataset_train.npz"))
    va = load(os.path.join(args.data, "dataset_val.npz"))
    te = load(os.path.join(args.data, "dataset_test.npz"))
    meta = json.load(open(os.path.join(args.data, "dataset_meta.json"), encoding="utf-8"))
    files = meta["files"]
    time_mode = meta.get("split_mode") == "time"
    print(f"Train {len(tr['y']):,} / Val {len(va['y']):,} / Test {len(te['y']):,}")
    if time_mode:
        print("시간 순서 분리 (같은 장비의 앞 시기=Train, 중간=Val, 뒤 시기=Test). 장비:", files)
        print("시기 경계(t_end 초):", meta["time_bounds"], "\n")
    else:
        print("장비 분리:", meta["splits"], "\n")

    neg_te = te["y"] == 0

    # ---------------- [A] ----------------
    print("=== [A] 전체 성능 (모든 이상 종류로 학습) ===")
    m_all = fit(tr["X"], tr["y"])
    p_te = score(m_all, te["X"])
    print(f"Test AUC = {f2(auc(te['y'], p_te))}")
    print("이상 종류별 AUC:", {nm: f2(auc(te["y"][neg_te | (te["atype"] == t)],
                                        p_te[neg_te | (te["atype"] == t)]))
                             for t, nm in enumerate(ATYPES)})
    print("변화 크기(eff)별 AUC:")
    curve = {}
    for t, nm in [(-2, "all")] + list(enumerate(ATYPES)):
        row = []
        for a, b in zip(EFF_BINS[:-1], EFF_BINS[1:]):
            ab = (te["y"] == 1) & (te["eff"] >= a) & (te["eff"] < b)
            if t >= 0:
                ab &= te["atype"] == t
            if ab.sum() < 20:
                row.append(float("nan"))
                continue
            mm = neg_te | ab
            row.append(auc(te["y"][mm], p_te[mm]))
        curve[nm] = row
    hdr = "  ".join(f"{a}-{b if b < 99 else 'inf'}" for a, b in zip(EFF_BINS[:-1], EFF_BINS[1:]))
    print(f"            구간: {hdr}")
    for nm, row in curve.items():
        print(f"  {nm:>10}: " + "   ".join(f2(v) for v in row))
    print("원인 센서(행)별 AUC:", {meta['sensors'][c]: f2(auc(te["y"][neg_te | (te["cause"] == c)],
                                                              p_te[neg_te | (te["cause"] == c)]))
                                for c in range(4)})

    # ---------------- [I] ----------------
    print("\n=== [I] 학습 데이터 양에 따른 성능 (학습 데이터를 일부만 쓸 때 Test AUC) ===")
    print("비율을 줄여도 AUC가 거의 안 변하면 데이터 양은 병목이 아님. 계속 오르면 데이터가 더 필요함")
    rs = np.random.default_rng(0)
    nt = len(tr["y"])
    for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
        k = max(200, int(nt * frac))
        sub = rs.choice(nt, k, replace=False) if k < nt else np.arange(nt)
        if len(np.unique(tr["y"][sub])) < 2:
            continue
        mf = fit(tr["X"][sub], tr["y"][sub])
        print(f"  학습 {k:>7,}개 ({frac:>4.0%}): Test AUC = {f2(auc(te['y'], score(mf, te['X'])))}")

    # ---------------- [B] ----------------
    print("\n=== [B] 종류 간 일반화: 행=학습에 쓴 이상 종류, 열=평가 이상 종류 (AUC) ===")
    print("대각선만 높고 비대각선이 0.5 근처면 주입 흔적(shortcut)을 외운 것")
    print(" " * 12 + "".join(f"{nm:>11}" for nm in ATYPES))
    for a, na in enumerate(ATYPES):
        mtr = (tr["y"] == 0) | (tr["atype"] == a)
        m = fit(tr["X"][mtr], tr["y"][mtr])
        p = score(m, te["X"])
        row = []
        for b in range(3):
            mm = neg_te | (te["atype"] == b)
            row.append(auc(te["y"][mm], p[mm]))
        print(f"{na:>12}" + "".join(f"{f2(v):>11}" for v in row))

    # ---------------- [C] ----------------
    print("\n=== [C] leave-one-type-out: 나머지 두 종류로 학습 -> 안 본 종류로 평가 ===")
    for b, nb in enumerate(ATYPES):
        mtr = (tr["y"] == 0) | ((tr["y"] == 1) & (tr["atype"] != b))
        m = fit(tr["X"][mtr], tr["y"][mtr])
        p = score(m, te["X"])
        mm = neg_te | (te["atype"] == b)
        print(f"  안 본 종류 {nb:>9}: AUC = {f2(auc(te['y'][mm], p[mm]))}")

    # ---------------- [E] ----------------
    print("\n=== [E] 실제 정상 윈도우 오탐률 ===")
    p_va = score(m_all, va["X"])
    yv = va["y"]
    qs = [0.5, 0.9, 0.95, 0.99]
    print("Val 정상 점수 분위수 :", {q: round(float(np.quantile(p_va[yv == 0], q)), 3) for q in qs})
    print("Val 이상 점수 분위수 :", {q: round(float(np.quantile(p_va[yv == 1], q)), 3) for q in qs})
    # 임계값: Val에서 TPR-FPR(Youden J)이 최대가 되는 지점 (분위수 방식은 점수가 1 근처로 포화되면 무의미해짐)
    cand = np.unique(np.quantile(p_va, np.linspace(0, 1, 501)))
    jv = [float((p_va[yv == 1] > t).mean() - (p_va[yv == 0] > t).mean()) for t in cand]
    thr = float(cand[int(np.argmax(jv))])
    print(f"임계값(Val Youden J) = {thr:.3f}")
    fpr = float((p_te[neg_te] > thr).mean())
    print(f"Test 정상 FPR = {fpr:.3f} (Specificity = {1 - fpr:.3f}), Test 이상 TPR = {float((p_te[~neg_te] > thr).mean()):.3f}")
    print("장비별 점검 (정상 점수 중앙값이 높으면 그 장비의 정상이 이상처럼 보이는 것 = 장비 간 분포 차이):")
    print("  장비     분리   정상 점수 중앙값   FPR@임계값   장비 내 AUC(정상 vs 이상)")
    for nm_, d_, p_ in (("val", va, p_va), ("test", te, p_te)):
        for f in np.unique(d_["file_id"]):
            mf = d_["file_id"] == f
            mn = mf & (d_["y"] == 0)
            if mn.sum() == 0:
                continue
            print(f"  {files[int(f)]}   {nm_:>4}   {float(np.median(p_[mn])):>8.3f}        "
                  f"{float((p_[mn] > thr).mean()):>7.3f}      {f2(auc(d_['y'][mf], p_[mf]))}   (정상 {int(mn.sum()):,}개)")
    # 장비 하나(예: 정상이 전부 이상으로 보이는 장비)가 임계값을 좌우하지 않도록, 장비별 TPR-FPR 평균이 최대인 임계값도 계산
    vt = np.unique(va["file_id"])
    jm = []
    for t_ in cand:
        js = []
        for f in vt:
            mf = va["file_id"] == f
            a1, a0 = mf & (yv == 1), mf & (yv == 0)
            if a1.sum() and a0.sum():
                js.append(float((p_va[a1] > t_).mean() - (p_va[a0] > t_).mean()))
        jm.append(np.mean(js))
    thr_m = float(cand[int(np.argmax(jm))])
    fpr_m = float((p_te[neg_te] > thr_m).mean())
    print(f"임계값(장비별 Youden J 평균, macro) = {thr_m:.3f} -> Test FPR = {fpr_m:.3f}, "
          f"Test TPR = {float((p_te[~neg_te] > thr_m).mean()):.3f}")
    print("참고(낙관적, Test 정상 분위수로 임계값을 정한 경우):")
    for q in (0.99, 0.95):
        t_ = float(np.quantile(p_te[neg_te], q))
        print(f"  FPR {1 - q:.0%} 에서 Test TPR = {float((p_te[~neg_te] > t_).mean()):.3f}")

    # ---------------- [E2] ----------------
    print("\n=== [E2] 장비별 정상 윈도우 센서 평균 (X_raw 단위, 모든 장비) — 어떤 장비가 왜 다른지 확인용 ===")
    print("  장비     분리   " + "  ".join(f"{s_[:12]:>12}" for s_ in meta["sensors"]))
    for nm_, d_ in (("train", tr), ("val", va), ("test", te)):
        for f in np.unique(d_["file_id"]):
            mn = (d_["file_id"] == f) & (d_["y"] == 0)
            if mn.sum() == 0:
                continue
            mu = d_["X_raw"][mn].mean(axis=(0, 2))
            print(f"  {files[int(f)]}   {nm_:>5}  " + "  ".join(f"{v:>12.3f}" for v in mu))

    # ---------------- [F] ----------------
    print("\n=== [F] shortcut 점검: 센서 관측 경계값(obs_min/obs_max)이 정확히 나타나는 윈도우 비율 ===")
    lo, hi = np.array(meta["obs_min"]), np.array(meta["obs_max"])
    for nm, d in [("train", tr), ("test", te)]:
        Xr = d["X_raw"]
        hit = np.zeros(len(Xr), bool)
        for i in range(4):
            hit |= np.isclose(Xr[:, i, :], lo[i], atol=1e-5).any(axis=1)
            hit |= np.isclose(Xr[:, i, :], hi[i], atol=1e-5).any(axis=1)
        print(f"  {nm}: 정상 {float(hit[d['y'] == 0].mean()):.3f}  이상 {float(hit[d['y'] == 1].mean()):.3f}"
              "   (두 값 차이가 크면 shortcut)")

    # ---------------- [G] ----------------
    print("\n=== [G] 원인 센서 복원 정확도 (이상 윈도우만, 4개 중 선택, 무작위 = 0.25) ===")
    ia = tr["y"] == 1
    mc = HistGradientBoostingClassifier(max_iter=150, random_state=0)
    mc.fit(feat(tr["X"][ia]), tr["cause"][ia])
    ta = te["y"] == 1
    pred = mc.predict(feat(te["X"][ta]))
    print(f"전체 정확도 = {float((pred == te['cause'][ta]).mean()):.3f}")
    for t, nm in enumerate(ATYPES):
        mm = te["atype"][ta] == t
        if mm.sum() > 20:
            print(f"  {nm}: {float((pred[mm] == te['cause'][ta][mm]).mean()):.3f}")

    # ---------------- [D] 그림 ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [f"{a}-{b if b < 99 else 'inf'}" for a, b in zip(EFF_BINS[:-1], EFF_BINS[1:])]
        plt.figure(figsize=(6, 4))
        for nm, row in curve.items():
            plt.plot(xs, row, "-o", label=nm)
        plt.axhline(0.5, color="gray", lw=0.8)
        plt.xlabel("anomaly magnitude (eff, in sensor std)")
        plt.ylabel("AUC (test period)" if time_mode else "AUC (test, unseen tools)")
        plt.legend()
        plt.tight_layout()
        pth = os.path.join(args.data, "detection_curve.png")
        plt.savefig(pth, dpi=130)
        print("\n[D] 크기별 탐지 곡선 저장:", pth)
    except Exception as e:  # noqa
        print("그림 생성 실패(무시 가능):", e)

    # ---------------- [H] ----------------
    if args.cache:
        print("\n=== [H] 부록: 실제 고장 직전 윈도우에 모델 적용 (Val+Test 장비의 윈도우만) ===")
        etch_only = bool(meta["params"].get("etch_only", False))
        print(f"비교 대상: Val+Test 정상 윈도우. 데이터셋이 etch-only 이면 고장 직전 윈도우도 식각 중만 사용 (etch_only={etch_only})")
        print("unmatched = 공정 조건 통제 없음 / matched = 같은 stage-recipe-step끼리만 비교한 층화 AUC")
        mean, std = np.array(meta["mean"], np.float32), np.array(meta["std"], np.float32)
        order = ["ETCHBEAMVOLTAGE", "IONGAUGEPRESSURE", "ETCHGASCHANNEL1READBACK", "FLOWCOOLPRESSURE"]
        xs_real, ttf, keys, tools = [], [], [], []
        for name in sorted(set(meta["splits"]["val"] + meta["splits"]["test"])):
            fp = os.path.join(args.cache, name + ".npz")
            if not os.path.exists(fp):
                continue
            z = np.load(fp)
            sens = [str(s_) for s_ in z["sensors"]]
            idx = [sens.index(s_) for s_ in order]
            X = np.transpose(z["Xw"][:, :, idx], (0, 2, 1)).astype(np.float32)
            ok = np.isfinite(X).all(axis=(1, 2)) & (z["ttf_any"] <= 9600)
            if etch_only:
                ok &= z["shut_frac"] == 1
            if time_mode:        # Val/Test 시기의 고장 직전 윈도우만 사용 (학습 시기 제외)
                ok &= z["t_end"] >= meta["time_bounds"][name]["val_start"]
            c = z["cond"]
            cc = np.nan_to_num(c, nan=0).astype(np.int64)
            k = np.where(z["cond_const"] & ~np.isnan(c).any(axis=1),
                         (cc[:, 0] * 1000 + cc[:, 1]) * 1000 + cc[:, 2], -1)
            Xo = X[ok]
            if meta["params"].get("per_tool_norm", False):
                # make_dataset 과 같은 방식: 그 장비의 정상풀 윈도우로 센서별 평균/표준편차를 구해 먼저 표준화
                P = meta["params"]
                norm_src = z["Xw"][:, :, idx]
                nm_ = (z["ttf_any"] >= P["normal_min"]) & (z["since"] >= P["since_min"]) \
                    & np.isfinite(norm_src).all(axis=(1, 2))
                if P.get("normal_local", 0) > 0:
                    nm_ &= np.minimum(z["ttf_any"], z["since"]) <= P["normal_local"]
                if P.get("etch_only", False):
                    nm_ &= z["shut_frac"] == 1
                if nm_.sum() > 100:
                    fl = np.transpose(norm_src[nm_], (2, 0, 1)).reshape(4, -1)
                    m_t = fl.mean(axis=1)
                    s_t = np.where(fl.std(axis=1) < 1e-6, 1.0, fl.std(axis=1))
                    Xo = (Xo - m_t[None, :, None]) / s_t[None, :, None]
            xs_real.append(((Xo - mean[None, :, None]) / std[None, :, None]).astype(np.float32))
            ttf.append(z["ttf_any"][ok])
            keys.append(k[ok])
            tools.append(np.full(int(ok.sum()), files.index(name)))
        if xs_real:
            Xr, tt, kr, tr_ = np.concatenate(xs_real), np.concatenate(ttf), np.concatenate(keys), np.concatenate(tools)
            if meta["params"].get("clip_z", 0) > 0:
                Xr = np.clip(Xr, -meta["params"]["clip_z"], meta["params"]["clip_z"])
            ref = [va, te]
            Xn = np.concatenate([d_["X"][d_["y"] == 0] for d_ in ref])
            kn = np.concatenate([d_["key"][d_["y"] == 0] for d_ in ref])
            tn = np.concatenate([d_["file_id"][d_["y"] == 0] for d_ in ref])
            pr, pn = score(m_all, Xr), score(m_all, Xn)
            print("matched = 같은 장비 + 같은 stage-recipe-step 의 정상과만 비교한 가중 평균 AUC")
            for a, b in [(0, 300), (300, 600), (600, 1200), (1200, 2400), (2400, 4800), (4800, 9600)]:
                mm = (tt >= a) & (tt < b)
                if mm.sum() < 20:
                    continue
                yy = np.r_[np.ones(int(mm.sum())), np.zeros(len(pn))]
                a_un = auc(yy, np.r_[pr[mm], pn])
                num, den = 0.0, 0
                for t_ in np.unique(tr_[mm]):
                    for kk_ in np.unique(kr[mm & (tr_ == t_)]):
                        if kk_ < 0:
                            continue
                        m1 = mm & (tr_ == t_) & (kr == kk_)
                        m0 = (tn == t_) & (kn == kk_)
                        if m1.sum() >= 10 and m0.sum() >= 10:
                            y2 = np.r_[np.ones(int(m1.sum())), np.zeros(int(m0.sum()))]
                            num += m1.sum() * auc(y2, np.r_[pr[m1], pn[m0]])
                            den += int(m1.sum())
                a_m = num / den if den else float("nan")
                print(f"  TTF {a}-{b}s: n={int(mm.sum()):,}  AUC unmatched={f2(a_un)}  "
                      f"matched={f2(a_m)} (매칭 가능 비율 {den / mm.sum():.0%})  "
                      f"이상 판정 비율={float((pr[mm] > thr).mean()):.3f} (정상 {float((pn > thr).mean()):.3f})")
        else:
            print("  Val/Test 장비의 cache 윈도우를 찾지 못했습니다.")
    print("\n끝.")


if __name__ == "__main__":
    main()