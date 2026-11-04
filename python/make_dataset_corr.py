#!/usr/bin/env python
"""
관계 이탈(corr_break) 합성 이상 + 어려운 음성(무해한 큰 변동) 데이터셋 생성기

make_dataset.py 와 같은 cache/분할/표준화 절차를 쓰되, 이상 유형만 바꾼 통제 실험용이다.
(make_dataset.py 는 건드리지 않는다. 같은 폴더에 두고 실행. make_dataset.py 의 load_base 를 재사용한다.)

설계
  정상 관계 구조: Train 정상 윈도우에서 센서 쌍 (i,j) 의 2x2 공분산을 구한다 (윈도우 x 시점 전체를 표본으로).
    - 소축(minor axis): 분산이 가장 작은 방향. 이 방향으로 움직이면 '두 센서가 평소의 상관을 깨는' 이동이다.
    - 대축(major axis): 분산이 가장 큰 방향. 이 방향으로 움직이면 '평소처럼 같이 움직이는' 이동이다.
  이상(label 1, corr_break): 선택한 쌍을 소축 방향으로 d x sqrt(lambda_min) 만큼 4시점 내내 이동 (d = 마할라노비스 거리, 기본 2~5)
      센서별 이동은 상관이 강한 쌍일수록 작다. 값은 정상 관측 범위 안에 머물러야 한다(벗어나면 버리고 다시 뽑음).
  어려운 음성(label 0, benign): 같은 방식으로 쌍을 대축 방향으로 m x sqrt(lambda_max) 이동 (m 기본 0.5~2.5)
      센서별 이동은 이상보다 크지만 정상 관계를 따라가는 '무해한 큰 변동'이다.
  -> 센서별 임계값 방식은 구조적으로 풀기 어렵고(센서별 이동이 작으면 더), 센서 간 관계를 쓰는 모델이 풀 수 있다.
  (기준선: 마할라노비스 = 정상만으로 피팅하는 비지도 기준, HGB = 지도 학습 기준. 둘 중 높은 쪽을 참고 상한으로 본다.)
  방향(부호)은 무작위. 원인 센서(cause) = 쌍 중 이동량(표준편차 대비)이 더 큰 센서, 쌍 번호는 pair 배열에 저장.

atype 코드: 0 = corr_break(이상), -1 = 일반 정상, -2 = 무해한 큰 변동(음성)

사용법 (make_dataset.py 와 같은 옵션)
  python make_dataset_corr.py --cache cache --out dataset_corr --tools 03_M02 09_M01 --etch-only --normal-local 0 \
      --per-file 60000 --clip-z 8 --obs-pct 0.1

필요 패키지: numpy, scikit-learn (sanity check 에만)
"""
import argparse
import json
import os
import sys

import numpy as np

import make_dataset as md

PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
ATYPES = ["corr_break"]


def pair_structure(Xtr_all):
    """Train 정상 윈도우(raw)에서 센서 쌍별 소축/대축 정보를 구한다."""
    pts = np.transpose(Xtr_all, (1, 0, 2)).reshape(4, -1).astype(np.float64)
    info = []
    for (i, j) in PAIRS:
        C = np.cov(pts[[i, j]])
        w, V = np.linalg.eigh(C)                                   # 오름차순
        rho = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])
        info.append(dict(pair=(i, j), rho=float(rho), lam_min=float(w[0]), lam_max=float(w[1]),
                         v_min=V[:, 0], v_max=V[:, 1],
                         minor_ratio=float(np.sqrt(w[0]) / (C[0, 0] * C[1, 1]) ** 0.25)))
    return info


def make_shifted(X, info, pair_ok, rng, kind, d_lo, d_hi, lo, hi, std, tries=10):
    """선택한 쌍을 소축(kind='minor') 또는 대축(kind='major') 방향으로 이동. 반환: Xo, pair, cause, eff, mag, ok"""
    N = len(X)
    Xo = X.copy()
    pair = np.full(N, -1, np.int64)
    cause = np.full(N, -1, np.int64)
    eff = np.zeros(N, np.float32)
    mag = np.zeros(N, np.float32)
    done = np.zeros(N, bool)
    for _ in range(tries):
        idx = np.flatnonzero(~done)
        if len(idx) == 0:
            break
        pp = rng.choice(pair_ok, len(idx))
        sgn = rng.choice([-1.0, 1.0], len(idx))
        dd = rng.uniform(d_lo, d_hi, len(idx))
        for p in np.unique(pp):
            m = pp == p
            ii = idx[m]
            i, j = info[p]["pair"]
            vec = info[p]["v_min"] if kind == "minor" else info[p]["v_max"]
            lam = info[p]["lam_min"] if kind == "minor" else info[p]["lam_max"]
            delta = (sgn[m] * dd[m])[:, None] * np.sqrt(lam) * vec[None, :]          # (n, 2) raw 단위
            ni = X[ii, i, :] + delta[:, 0:1]
            nj = X[ii, j, :] + delta[:, 1:2]
            ok = (((ni >= lo[i]) & (ni <= hi[i])).all(axis=1)) & (((nj >= lo[j]) & (nj <= hi[j])).all(axis=1))
            take = ii[ok]
            Xo[take, i, :] = ni[ok]
            Xo[take, j, :] = nj[ok]
            pair[take] = p
            mag[take] = dd[m][ok]
            ei = np.abs(delta[:, 0]) / std[i]
            ej = np.abs(delta[:, 1]) / std[j]
            eff[take] = ((ei + ej) / 2)[ok]
            cause[take] = np.where(ei >= ej, i, j)[ok]
            done[take] = True
    return Xo, pair, cause, eff, mag, done


# ---------------------------------------------------------------------------
# 기준선(학습 없음): exp_corr_break.py 에서도 재사용
# ---------------------------------------------------------------------------
def baseline_scores(tr, te, ridge=1e-3):
    """Train 정상(label 0, 무해한 변동 포함 X)으로 피팅한 기준선 점수들을 Test 에 대해 계산."""
    plain = (tr["y"] == 0) & (tr["atype"] == -1)
    Xp = tr["X"][plain].reshape(plain.sum(), 16).astype(np.float64)
    mu = Xp.mean(axis=0)
    C = np.cov(Xp.T)
    C = C + ridge * np.trace(C) / 16 * np.eye(16)
    Ci = np.linalg.inv(C)
    Xt = te["X"].reshape(len(te["X"]), 16).astype(np.float64)
    d = Xt - mu
    maha = np.einsum("ni,ij,nj->n", d, Ci, d)
    maxz = np.abs(te["X"]).max(axis=(1, 2))                          # 센서별 임계값 방식(가장 관대하게: 어느 한 값이라도 크면 이상)
    out = dict(maha=maha, maxz=maxz)
    # 엄격한 범위 점검(계획서의 Threshold 방식): 관측 범위를 벗어난 값이 있으면 이상. 합성 이상은 항상 범위 안이라 점수가 상수가 된다.
    out["range"] = np.zeros(len(Xt))
    return out


def auc_safe(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def report_scores(te, scores, title="Test"):
    """점수 하나에 대해 전체/일반 음성/무해한 변동 음성/마할라노비스 거리 구간별 AUC 를 계산."""
    y, at, mg = te["y"], te["atype"], te["maha"]
    anom = y == 1
    plain, benign = (y == 0) & (at == -1), (y == 0) & (at == -2)
    out = dict(all=auc_safe(y, scores), plain=auc_safe(y[anom | plain], scores[anom | plain]))
    out["benign"] = auc_safe(y[anom | benign], scores[anom | benign]) if benign.sum() > 20 else float("nan")
    for lo_, hi_ in ((2, 3), (3, 4), (4, 5.01)):
        sel = anom & (mg >= lo_) & (mg < hi_)
        mm = sel | (y == 0)
        out[f"d{lo_}-{int(hi_)}"] = auc_safe(y[mm], scores[mm]) if sel.sum() > 20 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="dataset_corr")
    ap.add_argument("--per-file", type=int, default=60000)
    ap.add_argument("--normal-min", type=float, default=86400)
    ap.add_argument("--since-min", type=float, default=21600)
    ap.add_argument("--normal-local", type=float, default=0)
    ap.add_argument("--etch-only", action="store_true")
    ap.add_argument("--tools", nargs="*", default=None)
    ap.add_argument("--split", choices=["time"], default="time")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--time-gap", type=float, default=86400)
    ap.add_argument("--per-tool-norm", action="store_true")
    ap.add_argument("--clip-z", type=float, default=0)
    ap.add_argument("--obs-pct", type=float, default=0)
    ap.add_argument("--min-corr", type=float, default=0.5, help="이 값 이상의 |상관|을 가진 센서 쌍만 사용 (없으면 상위 2쌍)")
    ap.add_argument("--d-lo", type=float, default=2.0, help="이상: 소축 마할라노비스 거리 하한")
    ap.add_argument("--d-hi", type=float, default=5.0)
    ap.add_argument("--m-lo", type=float, default=0.5, help="무해한 변동: 대축 거리 하한")
    ap.add_argument("--m-hi", type=float, default=2.5)
    ap.add_argument("--benign-frac", type=float, default=0.3, help="음성 중 무해한 큰 변동으로 바꿀 비율")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("정상 윈도우 불러오는 중...")
    base = md.load_base(args.cache, args, rng)
    names = sorted(base)
    parts = {"train": [], "val": [], "test": []}
    print(f"\n시간 순서 분리 (장비별 앞 시기=Train, 중간=Val, 뒤 시기=Test, 경계 gap {args.time_gap:.0f}초 제외)")
    for n in names:
        X_, k_, t_ = base[n]
        b1 = float(np.quantile(t_, 1 - args.val_frac - args.test_frac))
        b2 = float(np.quantile(t_, 1 - args.test_frac))
        g = args.time_gap
        masks = {"train": t_ < b1 - g, "val": (t_ >= b1) & (t_ < b2 - g), "test": t_ >= b2}
        for s_, m_ in masks.items():
            parts[s_].append((X_[m_], k_[m_], names.index(n), t_[m_]))
        print(f"  {n}: train {int(masks['train'].sum()):,} / val {int(masks['val'].sum()):,} / test {int(masks['test'].sum()):,}")
    for s_ in parts:
        if sum(len(p_[0]) for p_ in parts[s_]) < 20:
            sys.exit(f"{s_} 윈도우가 너무 적습니다.")

    Xtr_all = np.concatenate([p_[0] for p_ in parts["train"]])
    flat = np.transpose(Xtr_all, (1, 0, 2)).reshape(4, -1)
    mean = flat.mean(axis=1).astype(np.float32)
    std = np.where(flat.std(axis=1) < 1e-6, 1.0, flat.std(axis=1)).astype(np.float32)
    if args.obs_pct > 0:
        lo = np.percentile(flat, args.obs_pct, axis=1).astype(np.float32)
        hi = np.percentile(flat, 100 - args.obs_pct, axis=1).astype(np.float32)
    else:
        lo, hi = flat.min(axis=1).astype(np.float32), flat.max(axis=1).astype(np.float32)

    info = pair_structure(Xtr_all)
    print("\n센서 쌍의 정상 관계 구조 (Train 정상, 행 순서 = " + ", ".join(md.REPORT_ORDER) + ")")
    print(f"{'쌍':<12}{'상관':>8}{'소축 표준편차/기하평균σ':>26}{'대축/소축 분산비':>18}{'사용':>6}")
    qual = [p for p, d_ in enumerate(info) if abs(d_["rho"]) >= args.min_corr]
    if len(qual) == 0:
        qual = list(np.argsort([-abs(d_["rho"]) for d_ in info])[:2])
        print(f"[경고] |상관| >= {args.min_corr} 인 쌍이 없어 상위 2쌍을 사용합니다. 이 경우 '관계 이탈'과 '센서 하나의 이동'의 차이가 작아집니다.")
    for p, d_ in enumerate(info):
        i, j = d_["pair"]
        print(f"{md.REPORT_ORDER[i][:6] + '-' + md.REPORT_ORDER[j][:6]:<12}{d_['rho']:>8.2f}{d_['minor_ratio']:>26.3f}"
              f"{d_['lam_max'] / max(d_['lam_min'], 1e-12):>18.1f}{('O' if p in qual else '-'):>6}")
    pair_ok = np.array(qual)

    data = {}
    for split in ("train", "val", "test"):
        ps = parts[split]
        Xs = np.concatenate([p_[0] for p_ in ps])
        ks = np.concatenate([p_[1] for p_ in ps])
        fid = np.concatenate([np.full(len(p_[0]), p_[2]) for p_ in ps])
        ts = np.concatenate([p_[3] for p_ in ps])
        order = rng.permutation(len(Xs))
        Xs, ks, fid, ts = Xs[order], ks[order], fid[order], ts[order]
        h = len(Xs) // 2
        Xn, kn, fn, tn = Xs[:h].copy(), ks[:h], fid[:h], ts[:h]
        Xa, ka, fa, ta = Xs[h:], ks[h:], fid[h:], ts[h:]

        # 어려운 음성: 일부 정상 윈도우를 대축 방향으로 이동
        nb = int(round(args.benign_frac * len(Xn)))
        atn = np.full(len(Xn), -1, np.int64)
        pn = np.full(len(Xn), -1, np.int64)
        mgn = np.zeros(len(Xn), np.float32)
        efn = np.zeros(len(Xn), np.float32)
        if nb > 0:
            Xb, pb, cb, eb, mb, okb = make_shifted(Xn[:nb], info, pair_ok, rng, "major", args.m_lo, args.m_hi, lo, hi, std)
            sel = np.flatnonzero(okb)
            Xn[sel] = Xb[sel]
            atn[sel], pn[sel], mgn[sel], efn[sel] = -2, pb[sel], mb[sel], eb[sel]

        # 이상: 소축 방향 이동 (범위를 벗어나 끝내 실패한 것은 제외)
        Xa2, pa, ca, ea, ma, oka = make_shifted(Xa, info, pair_ok, rng, "minor", args.d_lo, args.d_hi, lo, hi, std)
        Xa2, pa, ca, ea, ma, fa, ka, ta = Xa2[oka], pa[oka], ca[oka], ea[oka], ma[oka], fa[oka], ka[oka], ta[oka]

        Xr = np.concatenate([Xn, Xa2])
        y = np.r_[np.zeros(len(Xn)), np.ones(len(Xa2))].astype(np.int64)
        at = np.r_[atn, np.zeros(len(Xa2), np.int64)]
        cs = np.r_[np.full(len(Xn), -1), ca].astype(np.int64)
        pr = np.r_[pn, pa].astype(np.int64)
        mg = np.r_[mgn, ma].astype(np.float32)
        ef = np.r_[efn, ea].astype(np.float32)
        fi = np.r_[fn, fa]
        kk = np.r_[kn, ka]
        tt = np.r_[tn, ta]
        o = rng.permutation(len(Xr))
        Xr, y, at, cs, pr, mg, ef, fi, kk, tt = Xr[o], y[o], at[o], cs[o], pr[o], mg[o], ef[o], fi[o], kk[o], tt[o]
        X = ((Xr - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
        if args.clip_z > 0:
            X = np.clip(X, -args.clip_z, args.clip_z)
        data[split] = dict(X=X, X_raw=Xr, y=y, cause=cs, atype=at, pair=pr, maha=mg, eff=ef, file_id=fi, key=kk,
                           t_end=tt, X_q88=md.q88(X))
        np.savez_compressed(os.path.join(args.out, f"dataset_{split}.npz"), **data[split])
        print(f"[{split}] N={len(y):,} (일반 정상 {int(((y == 0) & (at == -1)).sum()):,} / 무해한 큰 변동 {int((at == -2).sum()):,} / "
              f"이상 {int((y == 1).sum()):,}), 이상 주입 실패로 제외 {int(len(Xa) - len(Xa2)):,}")

    meta = dict(sensors=md.REPORT_ORDER, layout="X[sample, sensor, time] = (N, 4, 4)", mean=mean.tolist(), std=std.tolist(),
                obs_min=lo.tolist(), obs_max=hi.tolist(), atypes=ATYPES,
                atype_codes={"corr_break": 0, "plain_normal": -1, "benign_major_shift": -2},
                pairs=[dict(pair=list(d_["pair"]), rho=d_["rho"], lam_min=d_["lam_min"], lam_max=d_["lam_max"]) for d_ in info],
                used_pairs=[int(p) for p in pair_ok], files=names, params=vars(args))
    with open(os.path.join(args.out, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # sanity: 학습 없는 기준선들이 의도대로 동작하는지 (센서별 임계값은 실패, 마할라노비스는 성공해야 함)
    tr, te = data["train"], data["test"]
    sc = baseline_scores(tr, te)
    print("\n[sanity] 기준선 Test AUC  (전체 / 일반 음성만 / 무해한 큰 변동 음성만 / 이상의 마할라노비스 거리 구간별)")
    for nm, key in (("센서별 임계값(max|z|)", "maxz"), ("엄격한 범위 점검", "range"), ("마할라노비스 16차원(정상만으로 피팅)", "maha")):
        r = report_scores(te, sc[key])
        print(f"  {nm:<26} 전체 {r['all']:.3f} / 일반 {r['plain']:.3f} / 무해 {r['benign']:.3f} / "
              f"d2-3 {r['d2-3']:.3f}, d3-4 {r['d3-4']:.3f}, d4-5 {r['d4-5']:.3f}")
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        def feat(X):
            n = len(X)
            pair = [X[:, i, :].mean(axis=1) - X[:, j, :].mean(axis=1) for i in range(4) for j in range(i + 1, 4)]
            return np.c_[X.reshape(n, -1), np.stack(pair, axis=1)]
        m = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(feat(tr["X"]), tr["y"])
        r = report_scores(te, m.predict_proba(feat(te["X"]))[:, 1])
        print(f"  {'비선형 baseline(HGB)':<26} 전체 {r['all']:.3f} / 일반 {r['plain']:.3f} / 무해 {r['benign']:.3f} / "
              f"d2-3 {r['d2-3']:.3f}, d3-4 {r['d3-4']:.3f}, d4-5 {r['d4-5']:.3f}")
    except Exception as e:  # noqa
        print("HGB 생략:", e)
    big = {s: float((np.abs(data[s]['X']) > 8).any(axis=(1, 2)).mean()) for s in data}
    print(f"\n표준화 후 |x|>8 인 윈도우 비율 {big} (clip_z={args.clip_z})")
    print(f"완료. 출력 폴더: {args.out}")


if __name__ == "__main__":
    main()