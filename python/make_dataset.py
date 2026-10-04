#!/usr/bin/env python
"""
합성 이상 주입 학습 데이터 생성기

phm_diag.py build 로 만든 cache 의 '정상풀' 윈도우를 기반으로, 센서 4개 x 시간 4스텝 입력 행렬
(N, 4, 4) = [샘플, 센서, 시간] 과 라벨(정상/이상), 원인 센서, 이상 종류를 만든다.

핵심 설계
  - 실제 고장 라벨을 쓰지 않는다. (PHM 2018 분석 결과: 12초 윈도우 단위의 전조 구분 근거 없음)
  - 정상풀 윈도우 중 절반은 그대로 정상, 나머지 절반은 한 센서 행만 바꿔 이상으로 만든다.
    -> 정상과 이상이 같은 분포의 윈도우에서 나오므로 공정 조건/시기 혼입이 없다.
  - 장비(파일) 단위로 Train/Val/Test 분리, 표준화 통계는 Train 정상 윈도우로만 계산.
  - 이상 종류 (모두 센서 값이 해당 센서의 정상 관측 범위 안에 머물도록 clip)
      0 offset    : 한 센서 행 전체를 +-(mag x 표준편차) 만큼 이동
      1 ramp      : 4스텝에 걸쳐 0 -> +-(mag x 표준편차) 로 서서히 이탈
      2 swap_cond : 한 센서 행을 '다른 공정 조건'의 정상 윈도우의 같은 센서 행으로 교체
                    (값은 정상 범위인데 다른 센서와의 관계가 깨짐)
  - 실제로 바뀐 크기(eff = 평균|변화|/표준편차)가 --min-eff 미만이거나 값이 범위 밖으로 잘리면
    (경계값에 붙는 흔적이 생김) 재시도, 계속 실패하면 제외
  - --etch-only 로 식각 중 윈도우만 사용 가능 (shutter 값 2,3,255의 의미는 확인되지 않아 1만 열림으로 취급)

사용법
  python make_dataset.py --cache cache --out dataset
  (cache 는 phm_diag.py build 로 만든 폴더, 장비 단위 분리는 장비 파일이 3개 이상 필요)
  특정 장비 + 시간 순서 분리:
  python make_dataset.py --cache cache --out dataset_t --tools 03_M02 09_M01 --split time --etch-only --normal-local 0 --per-file 60000

필요 패키지: numpy, scikit-learn (마지막 sanity check에만 사용)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPORT_ORDER = ["ETCHBEAMVOLTAGE", "IONGAUGEPRESSURE",
                "ETCHGASCHANNEL1READBACK", "FLOWCOOLPRESSURE"]   # 중간보고서 SENSOR_COLS 순서
ATYPES = ["offset", "ramp", "swap_cond"]


def load_base(cache, args, rng):
    """장비 파일별 정상 윈도우 후보를 읽어 (X_raw, key, file_name) 리스트로 반환."""
    files = sorted(glob.glob(os.path.join(cache, "*.npz")))
    if args.tools:
        have = {os.path.basename(f)[:-4] for f in files}
        miss = [t for t in args.tools if t not in have]
        if miss:
            sys.exit(f"cache 에 없는 장비: {miss} (phm_diag.py build 로 먼저 만드세요)")
        files = [f for f in files if os.path.basename(f)[:-4] in set(args.tools)]
    elif len(files) < 3 and args.split == "tool":
        sys.exit("장비 단위 분리는 장비 파일이 3개 이상 필요합니다. (cache 폴더 확인)")
    out = {}
    for f in files:
        name = os.path.basename(f)[:-4]
        z = np.load(f)
        sens = [str(s) for s in z["sensors"]]
        idx = [sens.index(s) for s in REPORT_ORDER]
        Xw = z["Xw"][:, :, idx]                                    # (m, 4time, 4sensor)
        X = np.transpose(Xw, (0, 2, 1)).astype(np.float32)         # (m, 4sensor, 4time)
        normal = (z["ttf_any"] >= args.normal_min) & (z["since"] >= args.since_min)
        if args.normal_local > 0:
            normal &= np.minimum(z["ttf_any"], z["since"]) <= args.normal_local
        ok = normal & np.isfinite(X).all(axis=(1, 2))
        if args.etch_only:
            ok &= z["shut_frac"] == 1          # 4스텝 모두 shutter==열림(build의 --shutter-open 값)
        c = z["cond"]
        cc = np.nan_to_num(c, nan=0).astype(np.int64)
        key = np.where(z["cond_const"] & ~np.isnan(c).any(axis=1),
                       (cc[:, 0] * 1000 + cc[:, 1]) * 1000 + cc[:, 2], -1)
        sel = np.flatnonzero(ok)
        if len(sel) > args.per_file:
            sel = rng.choice(sel, args.per_file, replace=False)
        Xs = X[sel]
        if args.per_tool_norm:
            # 장비별 표준화: 그 장비 정상 윈도우의 센서별 평균/표준편차로 맞춤 (라벨을 쓰지 않는 보정)
            flat_t = np.transpose(Xs, (1, 0, 2)).reshape(4, -1)
            m_t, s_t = flat_t.mean(axis=1), flat_t.std(axis=1)
            s_t = np.where(s_t < 1e-6, 1.0, s_t)
            Xs = ((Xs - m_t[None, :, None]) / s_t[None, :, None]).astype(np.float32)
        out[name] = (Xs, key[sel], z["t_end"][sel])
        print(f"  {name}: 정상 후보 {int(ok.sum()):,} 중 {len(sel):,}개 사용")
    return out


def inject(X, key, mean, std, lo, hi, rng, mag_lo, mag_hi, min_eff, tries=10):
    """X (N,4,4) raw 윈도우에 이상을 주입. 반환: X_new, cause, atype, eff, keep(bool)"""
    N = len(X)
    Xo = X.copy()
    cause = rng.integers(0, 4, N)
    atype = rng.integers(0, 3, N)
    eff = np.zeros(N, np.float32)
    done = np.zeros(N, bool)
    ramp = np.linspace(0, 1, 4).astype(np.float32)
    for _ in range(tries):
        idx = np.flatnonzero(~done)
        if len(idx) == 0:
            break
        atype[idx] = rng.integers(0, 3, len(idx))
        c = cause[idx]
        orig = X[idx, c, :]                                        # (n, 4time)
        new = orig.copy()
        sgn = rng.choice([-1.0, 1.0], len(idx)).astype(np.float32)
        mag = rng.uniform(mag_lo, mag_hi, len(idx)).astype(np.float32)
        delta = (sgn * mag * std[c])[:, None]
        m0 = atype[idx] == 0
        new[m0] = orig[m0] + delta[m0]
        m1 = atype[idx] == 1
        new[m1] = orig[m1] + delta[m1] * ramp[None, :]
        m2 = atype[idx] == 2
        if m2.any():
            sub = idx[m2]
            j = rng.integers(0, N, len(sub))
            for _r in range(5):                                    # 같은 조건 donor는 다시 뽑기
                bad = (key[j] == key[sub]) & (key[sub] >= 0)
                if not bad.any():
                    break
                j[bad] = rng.integers(0, N, int(bad.sum()))
            new[m2] = X[j, c[m2], :]
        clipped_new = np.clip(new, lo[c][:, None], hi[c][:, None])
        was_clipped = np.abs(clipped_new - new).max(axis=1) > 1e-6
        new = clipped_new
        e = (np.abs(new - orig).mean(axis=1) / std[c]).astype(np.float32)
        ok = (e >= min_eff) & ~was_clipped      # 범위 밖으로 잘린 샘플은 버리고 다시 뽑음
        take = idx[ok]
        Xo[take, c[ok], :] = new[ok]
        eff[take] = e[ok]
        done[take] = True
    return Xo, cause, atype, eff, done


def q88(x):
    return np.clip(np.round(x * 256.0), -32768, 32767).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--per-file", type=int, default=6000, help="장비 파일당 사용할 정상 윈도우 수")
    ap.add_argument("--normal-min", type=float, default=86400)
    ap.add_argument("--since-min", type=float, default=21600)
    ap.add_argument("--normal-local", type=float, default=259200,
                    help="0이면 끔. 가장 가까운 고장과 N초 이내 시기의 정상만 사용")
    ap.add_argument("--etch-only", action="store_true",
                    help="식각 중(shutter 열림) 윈도우만 사용. 대기 상태의 평탄한 윈도우 제외")
    ap.add_argument("--tools", nargs="*", default=None,
                    help="사용할 장비 이름들 (예: 03_M02 09_M01). 지정 안 하면 cache 의 전체 장비")
    ap.add_argument("--split", choices=["tool", "time"], default="tool",
                    help="tool: 장비 단위 분리(기본). time: 장비별로 앞 시기=Train, 중간=Val, 뒤 시기=Test")
    ap.add_argument("--val-frac", type=float, default=0.15, help="--split time 일 때 Val 비율(시기 기준)")
    ap.add_argument("--test-frac", type=float, default=0.15, help="--split time 일 때 Test 비율(시기 기준)")
    ap.add_argument("--time-gap", type=float, default=86400,
                    help="--split time 일 때 분리 경계에서 버리는 시간(초). 인접 윈도우 누수 방지 (기본 1일)")
    ap.add_argument("--per-tool-norm", action="store_true",
                    help="장비별로 먼저 표준화(장비 간 정상 분포 차이 보정). 실제 사용 시 PC 쪽에서 장비별 보정값이 필요함. "
                         "이 경우 X_raw 는 '장비별 표준화된 값'이 됨")
    ap.add_argument("--clip-z", type=float, default=0,
                    help="0이면 끔. 표준화된 입력 X를 [-C, C]로 자름 (Q8.8 포화/오버플로 방지용, X_raw는 그대로)")
    ap.add_argument("--obs-pct", type=float, default=0,
                    help="0이면 센서 관측 범위를 min/max로. 예: 0.1 이면 0.1~99.9 백분위수 (스파이크가 범위를 넓히는 것 방지)")
    ap.add_argument("--mag-lo", type=float, default=0.5)
    ap.add_argument("--mag-hi", type=float, default=1.5)
    ap.add_argument("--min-eff", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-vectors", type=int, default=64, help="RTL 테스트 벡터로 내보낼 Test 윈도우 수")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("정상 윈도우 불러오는 중...")
    base = load_base(args.cache, args, rng)
    names = sorted(base)
    time_bounds = None
    if args.split == "tool":
        # 장비 분리는 데이터 내용과 무관하게 seed로만 결정 (--etch-only 등 옵션이 달라도 같은 분리 유지)
        perm = np.random.default_rng(args.seed).permutation(len(names))
        n_test = max(1, int(round(0.15 * len(names))))
        n_val = max(1, int(round(0.15 * len(names))))
        split_of = {}
        for r, i in enumerate(perm):
            split_of[names[i]] = "test" if r < n_test else ("val" if r < n_test + n_val else "train")
        groups = {s_: [n for n in names if split_of[n] == s_] for s_ in ("train", "val", "test")}
        print("\n장비 파일 분리:", {k: v for k, v in groups.items()})
        parts = {s_: [(base[n][0], base[n][1], names.index(n), base[n][2]) for n in groups[s_]]
                 for s_ in ("train", "val", "test")}
    else:
        groups = {s_: list(names) for s_ in ("train", "val", "test")}
        parts = {"train": [], "val": [], "test": []}
        time_bounds = {}
        print(f"\n시간 순서 분리 (장비별로 앞 시기=Train, 중간=Val, 뒤 시기=Test, 경계 gap {args.time_gap:.0f}초 제외)")
        for n in names:
            X_, k_, t_ = base[n]
            b1 = float(np.quantile(t_, 1 - args.val_frac - args.test_frac))
            b2 = float(np.quantile(t_, 1 - args.test_frac))
            g = args.time_gap
            masks = {"train": t_ < b1 - g, "val": (t_ >= b1) & (t_ < b2 - g), "test": t_ >= b2}
            for s_, m_ in masks.items():
                parts[s_].append((X_[m_], k_[m_], names.index(n), t_[m_]))
            time_bounds[n] = dict(val_start=b1, test_start=b2, gap=g)
            print(f"  {n}: train {int(masks['train'].sum()):,} / val {int(masks['val'].sum()):,} "
                  f"/ test {int(masks['test'].sum()):,} 윈도우")
            trk = k_[masks["train"]]
            trk = np.unique(trk[trk >= 0])
            for s_ in ("val", "test"):
                kk_ = k_[masks[s_]]
                kk_ = kk_[kk_ >= 0]
                if len(kk_):
                    print(f"    {s_}: 학습 시기에 없던 공정 조건(stage/recipe/step)의 윈도우 비율 = "
                          f"{float((~np.isin(kk_, trk)).mean()):.3f}")
    for s_ in ("train", "val", "test"):
        if sum(len(p_[0]) for p_ in parts[s_]) < 20:
            sys.exit(f"{s_} 윈도우가 너무 적습니다. --per-file 을 늘리거나 --normal-local 0 으로 풀어 보세요.")

    # Train 정상 윈도우로 표준화 통계 및 센서별 관측 범위 계산
    Xtr_all = np.concatenate([p_[0] for p_ in parts["train"]])           # (M, 4, 4)
    flat = np.transpose(Xtr_all, (1, 0, 2)).reshape(4, -1)
    mean = flat.mean(axis=1).astype(np.float32)
    std = flat.std(axis=1).astype(np.float32)
    if args.obs_pct > 0:
        lo = np.percentile(flat, args.obs_pct, axis=1).astype(np.float32)
        hi = np.percentile(flat, 100 - args.obs_pct, axis=1).astype(np.float32)
    else:
        lo = flat.min(axis=1).astype(np.float32)
        hi = flat.max(axis=1).astype(np.float32)
    p_lo = np.percentile(flat, 0.1, axis=1)
    p_hi = np.percentile(flat, 99.9, axis=1)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    print("\nTrain 센서 통계 (행 순서 = " + ", ".join(REPORT_ORDER) + ")")
    for i, s in enumerate(REPORT_ORDER):
        print(f"  {s}: mean={mean[i]:.3f} std={std[i]:.3f} range=[{lo[i]:.2f}, {hi[i]:.2f}] "
              f"(실제 min/max=[{flat[i].min():.2f}, {flat[i].max():.2f}], 0.1~99.9%=[{p_lo[i]:.2f}, {p_hi[i]:.2f}])")

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
        Xn, kn, fn, tn = Xs[:h], ks[:h], fid[:h], ts[:h]
        Xa, ka, fa, ta = Xs[h:], ks[h:], fid[h:], ts[h:]
        Xa2, cause, atype, eff, ok = inject(Xa, ka, mean, std, lo, hi, rng,
                                            args.mag_lo, args.mag_hi, args.min_eff)
        Xa2, cause, atype, eff, fa, ka, ta = (Xa2[ok], cause[ok], atype[ok], eff[ok],
                                              fa[ok], ka[ok], ta[ok])
        Xr = np.concatenate([Xn, Xa2])
        y = np.r_[np.zeros(len(Xn)), np.ones(len(Xa2))].astype(np.int64)
        cs = np.r_[np.full(len(Xn), -1), cause].astype(np.int64)
        at = np.r_[np.full(len(Xn), -1), atype].astype(np.int64)
        ef = np.r_[np.zeros(len(Xn)), eff].astype(np.float32)
        fi = np.r_[fn, fa]
        kk = np.r_[kn, ka]
        tt = np.r_[tn, ta]
        o = rng.permutation(len(Xr))
        Xr, y, cs, at, ef, fi, kk, tt = Xr[o], y[o], cs[o], at[o], ef[o], fi[o], kk[o], tt[o]
        X = ((Xr - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
        big = float((np.abs(X) > 8).any(axis=(1, 2)).mean())
        if args.clip_z > 0:
            X = np.clip(X, -args.clip_z, args.clip_z)
        print(f"[{split}] 표준화 후 |x|>8 인 윈도우 비율 = {big:.4f}" +
              (f" (clip_z={args.clip_z} 적용)" if args.clip_z > 0 else " (clip 안 함)"))
        data[split] = dict(X=X, X_raw=Xr, y=y, cause=cs, atype=at, eff=ef, file_id=fi, key=kk,
                           t_end=tt, X_q88=q88(X))
        np.savez_compressed(os.path.join(args.out, f"dataset_{split}.npz"), **data[split])
        print(f"[{split}] N={len(y):,} (정상 {int((y == 0).sum()):,} / 이상 {int((y == 1).sum()):,}), "
              f"주입 실패로 제외 {int(len(Xa) - len(Xa2)):,}")

    meta = dict(sensors=REPORT_ORDER, layout="X[sample, sensor, time] = (N, 4, 4)",
                mean=mean.tolist(), std=std.tolist(), obs_min=lo.tolist(), obs_max=hi.tolist(),
                atypes=ATYPES, cause_index="0..3 = 센서 행 인덱스, -1 = 정상",
                splits=groups, files=names, params=vars(args),
                split_mode=args.split, time_bounds=time_bounds)
    with open(os.path.join(args.out, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Q8.8 범위 확인 + RTL 테스트 벡터 (행 우선: 센서0의 4스텝, 센서1의 4스텝, ...)
    allq = np.concatenate([data[s]["X"] for s in data])
    print(f"\n입력 표준화 값 범위: [{allq.min():.2f}, {allq.max():.2f}]  (Q8.8 표현 범위 약 -128~127.996)")
    te = data["test"]
    nv = min(args.n_vectors, len(te["y"]))
    with open(os.path.join(args.out, "test_vectors_q88.mem"), "w") as f:
        for i in range(nv):
            for v in te["X_q88"][i].reshape(-1):
                f.write(f"{int(v) & 0xFFFF:04x}\n")
    np.savetxt(os.path.join(args.out, "test_vectors_labels.csv"),
               np.c_[te["y"][:nv], te["cause"][:nv], te["atype"][:nv]],
               fmt="%d", delimiter=",", header="label,cause_sensor,atype", comments="")
    print(f"RTL 테스트 벡터 {nv}개 저장 (16워드/윈도우, 행 우선 = 센서 x 시간 순서)")

    # sanity check: 비선형 baseline(HistGradientBoosting)이 합성 이상을 구분하는지 (모델 학습 전 확인용)
    # 주의: 이상 주입이 +-대칭이라 단순 선형 모델은 구조적으로 구분할 수 없음 (그래서 비선형 사용)
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score

        def feat(X):
            n = len(X)
            pair = [X[:, i, :].mean(axis=1) - X[:, j, :].mean(axis=1)
                    for i in range(4) for j in range(i + 1, 4)]
            return np.c_[X.reshape(n, -1), np.stack(pair, axis=1)]

        tr, te_ = data["train"], data["test"]
        m = HistGradientBoostingClassifier(max_iter=150, random_state=0)
        m.fit(feat(tr["X"]), tr["y"])
        p = m.predict_proba(feat(te_["X"]))[:, 1]
        print(f"\n[sanity] 비선형 baseline Test AUC = {roc_auc_score(te_['y'], p):.3f}")
        neg = te_["y"] == 0
        for t, nm in enumerate(ATYPES):
            mm = neg | (te_["atype"] == t)
            if (te_["atype"] == t).sum() > 20:
                print(f"  이상 종류 {nm}: AUC = {roc_auc_score(te_['y'][mm], p[mm]):.3f}")
        for a_, b_ in [(0.3, 0.7), (0.7, 1.2), (1.2, 99)]:
            sel_ab = (te_["eff"] >= a_) & (te_["eff"] < b_) & (te_["y"] == 1)
            if sel_ab.sum() > 20:
                mm = neg | sel_ab
                print(f"  변화 크기 eff {a_}~{b_}: AUC = {roc_auc_score(te_['y'][mm], p[mm]):.3f} (n={int(sel_ab.sum())})")
    except Exception as e:  # noqa
        print("sanity check 생략:", e)
    print(f"\n완료. 출력 폴더: {args.out}")


if __name__ == "__main__":
    main()