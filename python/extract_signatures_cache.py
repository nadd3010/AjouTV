#!/usr/bin/env python
"""
[분석] 우리 cache(phm_diag.py build 결과)에서 '모든' 실제 고장 event 의 신호 특성을 추출한다.

팀원 CSV 방식과 다른 점
  - 팀원은 '평균 이탈 점수로 관측 가능한 event 67개'만 골랐다. 여기서는 선별하지 않고 cache 에 있는 모든 고장 event 를 본다.
  - 같은 방식(조건별 정상 기준의 robust z RMS, 95백분위 초과 비율 >= 0.5)으로 '관측 가능' 여부를 계산해서
    전체 event 중 관측 가능한 것의 비율과, 관측 불가(크기로는 안 보이는) event 의 특성을 따로 볼 수 있다.
  - cache 는 우리 데이터셋과 같은 원본 단위라서 이후 합성 이상 생성기에 바로 연결된다.

event 구성
  - 한 event = 같은 파일의 같은 고장 시각(fault_time = t_end + ttf_any). 고장 직전 --horizon 초 안의 윈도우들.
  - 기준(정상): 같은 파일, 같은 공정 조건(stage/recipe/recipe_step)의 정상 풀 윈도우 (cache 의 정상 표본: 고장에서 86400초 이상 떨어진 곳)
  - Δ(4센서 x 4시점) = (event 윈도우 평균 - 정상 윈도우 평균) / 정상 표준편차(조건별, 파일 전체 표준편차의 5% 이상으로 바닥 처리)

사용법
  python extract_signatures_cache.py --cache cache --out sig_cache
  python extract_signatures_cache.py --cache cache --exclude-csv observable4_dataset_balanced.csv --out sig_cache
      (--exclude-csv: 팀원 CSV 의 event 를 제외. 팀원 event 를 나중에 독립 시험 데이터로 쓰려면 반드시 지정)
  python extract_signatures_cache.py --cache cache --only-csv observable4_dataset_balanced.csv --placebo-start 4800 --out sig_team
      (--only-csv: 팀원 CSV 의 event 만 분석. 팀원이 고른 event 의 이탈이 '고장 직전에만' 나타나는지, 먼 구간에서도 같은지 확인)

대조(placebo)  --placebo-start 4800
  - 각 event 의 '고장에서 먼 구간'(TTF 4800 ~ 4800+horizon 초)을 같은 절차로 분석해서, 고장 직전 구간과 나란히 비교한다.
  - 두 구간이 비슷하면 고장 직전 60초에 '고장 신호'라고 부를 만한 이탈이 따로 없다는 뜻이다 (정상 상태의 변동일 뿐).
  - 상관 변화처럼 잡음이 큰 지표도 같은 구조(연속 윈도우 3~4개)의 대조와 비교해야 해석할 수 있다.

주의
  - cache 윈도우는 4스텝씩 겹치지 않게 저장돼 고장 직전 60초에는 event 당 3~4개뿐이다. 평균 이동(방향, 크기)은 안정적이지만
    센서 간 상관 변화는 잡음이 크다. 필요하면 --horizon 120 처럼 넓힌다 (TTF 가 먼 윈도우가 섞여 크기는 작아질 수 있다).
출력: 터미널 요약 + <out>/fault_signatures_cache.json
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

SENSOR_ALL = ["IONGAUGEPRESSURE", "ETCHBEAMVOLTAGE", "ETCHGASCHANNEL1READBACK",
              "FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE", "ETCHSUPPRESSORCURRENT"]
HW = ["ETCHBEAMVOLTAGE", "IONGAUGEPRESSURE", "ETCHGASCHANNEL1READBACK", "FLOWCOOLPRESSURE"]
SHORT = ["BEAMV", "IONP", "GASCH1", "COOLP"]
FAULT_LABELS = ["pressure_low", "pressure_high", "leak", "unknown"]


def corr4(w):
    pts = w.transpose(1, 0, 2).reshape(w.shape[1], -1)
    return np.nan_to_num(np.corrcoef(pts))


def maha_score(Nr, Ev):
    """센서 간 '관계 이탈' 점수: 같은 조건 정상 윈도우(Nr)의 평균/공분산으로 계산한 event 윈도우(Ev)의 마할라노비스 거리^2 / 차원 수의 중앙값.
    정상이면 약 1. 공분산은 대각 쪽으로 수축(shrinkage)해서 정상 윈도우가 적을 때의 불안정을 줄인다.
    차원 4 = 윈도우 평균(센서 간 관계만), 16 = 시간 구조까지 포함."""
    k = Nr.shape[1]
    mu = Nr.mean(axis=0)
    C = np.cov(Nr.T)
    alpha = 0.2 if len(Nr) >= 300 else 0.5
    C = (1 - alpha) * C + alpha * np.diag(np.diag(C))
    C = C + 1e-8 * np.trace(C) / k * np.eye(k)
    d = Ev - mu
    return float(np.median(np.einsum("ni,ij,nj->n", d, np.linalg.inv(C), d)) / k)


def maha_parts(Nr, Ev):
    """4차원 관계 점수를 주축(정상 관계를 따라 움직이는 방향)과 나머지 축(관계 이탈)으로 나눈다. 정상이면 각각 약 1.
       반환: (전체, 주축 성분, 나머지 축 평균) 의 중앙값."""
    k = Nr.shape[1]
    mu = Nr.mean(axis=0)
    C = np.cov(Nr.T)
    alpha = 0.2 if len(Nr) >= 300 else 0.5
    C = (1 - alpha) * C + alpha * np.diag(np.diag(C))
    C = C + 1e-8 * np.trace(C) / k * np.eye(k)
    w, V = np.linalg.eigh(C)                                  # 오름차순
    zc = ((Ev - mu) @ V) / np.sqrt(w)[None, :]
    return (float(np.median((zc ** 2).sum(axis=1) / k)), float(np.median(zc[:, -1] ** 2)),
            float(np.median((zc[:, :-1] ** 2).mean(axis=1))))


def mcnemar(pre, pl):
    """짝지은 두 이진 판정의 McNemar 정확검정 (양측 이항). 반환: (pre만, 대조만, 둘 다, 둘 다 아님, p)"""
    from math import comb
    pre, pl = np.asarray(pre, bool), np.asarray(pl, bool)
    n10, n01 = int((pre & ~pl).sum()), int((~pre & pl).sum())
    n11, n00 = int((pre & pl).sum()), int((~pre & ~pl).sum())
    n = n10 + n01
    if n == 0:
        return n10, n01, n11, n00, 1.0
    k = min(n10, n01)
    pv = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n10, n01, n11, n00, pv


def robust_scores(Xn16, Xe16):
    """팀원 event_observability_analysis.py 와 같은 방식: median/MAD 기반 robust z 의 RMS."""
    center = np.median(Xn16, axis=0)
    mad = np.median(np.abs(Xn16 - center), axis=0)
    rs = 1.4826 * mad
    sd = Xn16.std(axis=0)
    scale = np.where(rs > 1e-4, rs, sd)
    scale = np.where(scale > 1e-4, scale, 1e-3)

    def sc(X):
        z = np.clip((X - center) / scale, -10, 10)
        return np.sqrt(np.mean(z ** 2, axis=1))
    sn = sc(Xn16)
    return sc(Xe16), np.percentile(sn, 95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="sig_cache")
    ap.add_argument("--horizon", type=float, default=60.0, help="고장 직전 몇 초까지를 event 윈도우로 볼지")
    ap.add_argument("--min-windows", type=int, default=2)
    ap.add_argument("--min-normal", type=int, default=30, help="같은 조건의 정상 윈도우가 이보다 적으면 event 제외")
    ap.add_argument("--normal-min", type=float, default=86400)
    ap.add_argument("--since-min", type=float, default=21600)
    ap.add_argument("--exclude-csv", default=None, help="팀원 CSV(file_id, event_time 열). 이 event 들은 제외")
    ap.add_argument("--only-csv", default=None, help="팀원 CSV(file_id, event_time 열). 이 event 들만 분석")
    ap.add_argument("--mark-csv", default=None, help="팀원 CSV. 모든 event 를 분석하되 이 CSV 의 event 에 표시만 한다 (표 [6] 에서 그룹별로 나눠 봄)")
    ap.add_argument("--placebo-start", type=float, default=0.0,
                    help="0 이면 사용 안 함. 예: 4800 -> TTF 4800~4800+horizon 구간을 대조(placebo)로 분석")
    ap.add_argument("--placebo-min-since", type=float, default=1800.0, help="대조 윈도우는 직전 고장 후 이만큼 지난 것만 사용")
    ap.add_argument("--shift-thr", type=float, default=0.5)
    ap.add_argument("--corr-thr", type=float, default=0.3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    def read_events(path):
        t = pd.read_csv(path)
        if "event_time" not in t.columns or "file_id" not in t.columns:
            raise SystemExit(f"{path} 에 file_id, event_time 열이 필요합니다.")
        t = t[t["label"] == 1] if "label" in t.columns else t
        out = set()
        for f, et in zip(t["file_id"].astype(str), pd.to_numeric(t["event_time"], errors="coerce")):
            if np.isfinite(et):
                out.add((f, int(round(et))))
        return out

    if a.exclude_csv and a.only_csv:
        raise SystemExit("--exclude-csv 와 --only-csv 는 함께 쓸 수 없습니다.")
    excl = read_events(a.exclude_csv) if a.exclude_csv else set()
    incl = read_events(a.only_csv) if a.only_csv else None
    mark = read_events(a.mark_csv) if a.mark_csv else set()
    if excl:
        print(f"제외할 팀원 event: {len(excl)}개")
    if incl is not None:
        print(f"분석할 팀원 event: {len(incl)}개 (cache 에서 실제로 찾아지는 개수는 아래 '추출' 줄에 나옵니다)")

    files = sorted(glob.glob(os.path.join(a.cache, "*.npz")))
    if not files:
        raise SystemExit("cache 에 .npz 가 없습니다. 먼저 phm_diag.py build 를 실행하세요.")
    hw_idx = [SENSOR_ALL.index(s) for s in HW]
    events, skipped = [], dict(no_cond=0, few_normal=0, few_windows=0, excluded=0, not_in_csv=0)
    for fpath in files:
        name = os.path.basename(fpath)[:-4]
        z = np.load(fpath)
        for k in ("t_end", "ttf_any", "ttf_k", "since", "Xw", "cond", "cond_const"):
            if k not in z.files:
                raise SystemExit(f"{name}: cache 에 '{k}' 가 없습니다 (phm_diag.py build 를 다시 실행).")
        t_end, ttf, ttfk, since = z["t_end"].astype(np.float64), z["ttf_any"], z["ttf_k"], z["since"]
        Xw = z["Xw"].astype(np.float64)                                   # (m, 4시점, 6센서)
        cond, cc = z["cond"], z["cond_const"]
        ok = cc & ~np.isnan(cond).any(axis=1)
        cn = np.nan_to_num(cond, nan=0).astype(np.int64)
        key = np.where(ok, (cn[:, 0] * 1000 + cn[:, 1]) * 1000 + cn[:, 2], -1)
        normal = (ttf >= a.normal_min) & (since >= a.since_min)
        if normal.sum() < a.min_normal:
            continue
        sd_file = Xw[normal].transpose(0, 2, 1).reshape(normal.sum(), len(SENSOR_ALL), 4).std(axis=(0, 2))
        pre = np.isfinite(ttf) & (ttf > 0) & (ttf <= a.horizon)
        ft = np.where(pre, np.rint(t_end + ttf), -1).astype(np.int64)          # 고장 직전 윈도우의 고장 시각
        ft_all = np.where(np.isfinite(ttf), np.rint(t_end + ttf), -1).astype(np.int64)  # 모든 윈도우의 다음 고장 시각

        def analyze(idx):
            """윈도우 인덱스 집합 -> 신호 특성 dict (조건 기준 불가 시 사유 문자열)."""
            if len(idx) < a.min_windows:
                return "few_windows"
            ks = key[idx][key[idx] >= 0]
            if len(ks) == 0:
                return "no_cond"
            dk = int(pd.Series(ks).mode().iloc[0])                          # 대표 공정 조건
            nm = np.flatnonzero(normal & (key == dk))
            if len(nm) < a.min_normal:
                return "few_normal"
            ev_all = Xw[idx].transpose(0, 2, 1)                           # (n, 6센서, 4시점)
            nr_all = Xw[nm].transpose(0, 2, 1)
            sd = np.maximum(nr_all.std(axis=(0, 2)), 0.05 * sd_file)
            delta6 = (ev_all.mean(axis=0) - nr_all.mean(axis=0)) / sd[:, None]  # (6,4)
            Xe, Xn = ev_all[:, hw_idx, :], nr_all[:, hw_idx, :]             # (n,4,4) 센서 행 x 시점 열 (HW 센서 순서)
            se, thr = robust_scores(Xn.reshape(len(Xn), 16), Xe.reshape(len(Xe), 16))
            m4 = maha_parts(Xn.mean(axis=2), Xe.mean(axis=2))
            return dict(cond_key=dk, n_pos=len(idx), n_neg=len(nm), obs_frac=float(np.mean(se > thr)),
                        delta_z=delta6[hw_idx], delta_z_all6=delta6, dcorr=corr4(Xe) - corr4(Xn),
                        maha4=m4[0], maha4_major=m4[1], maha4_minor=m4[2],
                        maha16=maha_score(Xn.reshape(len(Xn), 16), Xe.reshape(len(Xe), 16)))

        for fv in np.unique(ft[pre]):
            idx = np.flatnonzero(ft == fv)
            if (name, int(fv)) in excl:
                skipped["excluded"] += 1
                continue
            if incl is not None and (name, int(fv)) not in incl:
                skipped["not_in_csv"] += 1
                continue
            r = analyze(idx)
            if isinstance(r, str):
                skipped[r] += 1
                continue
            kk = np.median(ttfk[idx], axis=0)                               # 고장 종류별 '다음 고장까지' 시간 (없으면 inf)
            ftype = int(np.argmin(kk)) if np.isfinite(kk).any() else 3     # 가장 가까운 고장의 종류
            ev = dict(file=name, fault_time=int(fv), fault=FAULT_LABELS[ftype], placebo=None, in_team=((name, int(fv)) in mark), **r)
            if a.placebo_start > 0:
                pidx = np.flatnonzero((ft_all == fv) & np.isfinite(ttf) & (ttf >= a.placebo_start)
                                      & (ttf < a.placebo_start + a.horizon) & (since >= a.placebo_min_since))
                pr = analyze(pidx)
                if not isinstance(pr, str):
                    ev["placebo"] = pr
            events.append(ev)
    print(f"event {len(events)}개 추출. 제외/건너뜀: {skipped}")
    if not events:
        raise SystemExit("추출된 event 가 없습니다. --horizon 을 늘리거나 --min-normal 을 낮춰 보세요.")

    for e in events:
        e["obs_class"] = "observable" if e["obs_frac"] >= 0.5 else ("borderline" if e["obs_frac"] >= 0.2 else "not_observable")
    df = pd.DataFrame([{k: e[k] for k in ("file", "fault", "obs_class", "n_pos")} for e in events])
    print("\n[1] 고장 종류 x 관측 가능성 (팀원 기준: 정상 95백분위 초과 윈도우 비율 >= 0.5 = observable)")
    ct = pd.crosstab(df["fault"], df["obs_class"], margins=True)
    print(ct.to_string())
    obs_all = int((df["obs_class"] == "observable").sum())
    print(f"  -> 전체 {len(df)}개 중 observable {obs_all}개 ({obs_all / len(df):.1%}). (팀원은 4센서에서 observable 67개를 사용)")

    iu = np.triu_indices(4, 1)
    print("\n[2] 고장 종류별 신호 특성 (HW 4센서, 단위: 정상 표준편차 배수)")
    for fault in [f for f in FAULT_LABELS if f in set(df["fault"])]:
        ev = [e for e in events if e["fault"] == fault]
        D = np.stack([e["delta_z"] for e in ev])
        m = D.mean(axis=2)
        print("\n" + "=" * 96)
        print(f"{fault}  (event {len(ev)}개: " + ", ".join(f"{c} {sum(1 for e in ev if e['obs_class'] == c)}" for c in ("observable", "borderline", "not_observable")) + ")")
        print("=" * 96)
        print(f"{'센서':<8}{'평균 이동 중앙값':>16}{'부호 일관성':>12}{'|이동| 중앙값':>14}{'|이동| 75%':>12}{'주된 센서 비율':>14}")
        dom = np.abs(m).argmax(axis=1)
        for s in range(4):
            med = np.median(m[:, s])
            cons = float(np.mean(np.sign(m[:, s]) == np.sign(med))) if med != 0 else float("nan")
            print(f"{SHORT[s]:<8}{med:>16.2f}{cons:>12.2f}{np.median(np.abs(m[:, s])):>14.2f}{np.percentile(np.abs(m[:, s]), 75):>12.2f}{float(np.mean(dom == s)):>14.2f}")
        srt = np.sort(np.abs(m), axis=1)
        single = float(np.mean(srt[:, -1] >= 2 * np.maximum(srt[:, -2], 1e-9)))
        nbig = (np.abs(m) > 1.0).sum(axis=1)
        dist = {int(k): int(v) for k, v in zip(*np.unique(nbig, return_counts=True))}
        print(f"  한 센서 중심(주된 센서가 둘째의 2배 이상)인 event 비율 {single:.2f}, |이동|>1σ 인 센서 수 분포 {dist}")
        shape = np.array([(D[i, dom[i], -1] - D[i, dom[i], 0]) / (abs(m[i, dom[i]]) + 1e-6) for i in range(len(ev))])
        print(f"  시간 형태 (t - 첫시점)/|평균이동|: 중앙값 {np.median(shape):+.2f}, 25~75% [{np.percentile(shape, 25):+.2f}, {np.percentile(shape, 75):+.2f}]")

    print("\n[3] 관측 가능 여부별: 크기 이탈 vs 상관 변화")
    print(f"{'구분':<16}{'event':>7}{'max|이동| 중앙값':>16}{'이동<%.1fσ' % a.shift_thr:>10}{'그중 상관변화>%.1f' % a.corr_thr:>16}{'max|Δ상관| 중앙값':>18}")
    for c in ("observable", "borderline", "not_observable"):
        ev = [e for e in events if e["obs_class"] == c]
        if not ev:
            continue
        mx = np.array([np.abs(e["delta_z"].mean(axis=1)).max() for e in ev])
        dcm = np.array([np.abs(e["dcorr"][iu]).max() for e in ev])
        small = mx < a.shift_thr
        print(f"{c:<16}{len(ev):>7}{np.median(mx):>16.2f}{int(small.sum()):>10}{int((small & (dcm > a.corr_thr)).sum()):>16}{np.median(dcm):>18.2f}")
    print("  (상관 변화는 event 당 윈도우가 적어 잡음이 큽니다. '그중 상관변화' 개수는 참고용으로만 보세요.)")

    if a.placebo_start > 0:
        pe = [e for e in events if e["placebo"] is not None]
        print(f"\n[4] 고장 직전 vs 대조(placebo: TTF {a.placebo_start:.0f}~{a.placebo_start + a.horizon:.0f}초) - 둘 다 계산된 event {len(pe)}개")
        if len(pe) < 10:
            print("   대조가 계산된 event 가 너무 적어 비교하지 않습니다 (--placebo-start 를 낮추거나 --placebo-min-since 를 낮춰 보세요).")
        else:
            def stats(D):
                mx = np.array([np.abs(d["delta_z"].mean(axis=1)).max() for d in D])
                dc = np.array([np.abs(d["dcorr"][iu]).max() for d in D])
                of = np.array([d["obs_frac"] for d in D])
                return mx, dc, of
            real = [e for e in pe]
            plac = [e["placebo"] for e in pe]
            mr, cr, orr = stats(real)
            mp, cp, op = stats(plac)
            q = lambda v: "/".join(f"{np.percentile(v, p):.2f}" for p in (25, 50, 75, 90))
            print(f"{'지표':<34}{'고장 직전':>22}{'대조(placebo)':>22}")
            print(f"{'max|이동| 25/50/75/90%':<34}{q(mr):>22}{q(mp):>22}")
            print(f"{'max|이동| > 1σ 인 event 비율':<34}{np.mean(mr > 1):>22.2f}{np.mean(mp > 1):>22.2f}")
            print(f"{'max|이동| < 0.5σ 인 event 비율':<34}{np.mean(mr < 0.5):>22.2f}{np.mean(mp < 0.5):>22.2f}")
            print(f"{'observable(95백분위 초과 >= 50%) 비율':<34}{np.mean(orr >= 0.5):>22.2f}{np.mean(op >= 0.5):>22.2f}")
            print(f"{'평균 초과 윈도우 비율':<34}{orr.mean():>22.2f}{op.mean():>22.2f}")
            print(f"{'max|Δ상관| 25/50/75/90%':<34}{q(cr):>22}{q(cp):>22}")
            p95 = np.percentile(mp, 95)
            print(f"\n  대조의 max|이동| 95백분위 = {p95:.2f}σ.  고장 직전 event 중 이를 넘는 비율 {np.mean(mr > p95):.2f} (신호가 전혀 없다면 약 0.05)")
            print(f"  event 별 짝 비교: 고장 직전 max|이동| > 대조 max|이동| 인 비율 {np.mean(mr > mp):.2f} (신호가 없다면 약 0.50)")
            for fault in [f for f in FAULT_LABELS if any(e['fault'] == f for e in pe)]:
                sel = np.array([e["fault"] == fault for e in pe])
                if sel.sum() >= 5:
                    print(f"    {fault:<14} event {int(sel.sum()):>3}개: 대조 95백분위 초과 비율 {np.mean(mr[sel] > p95):.2f}, 짝 비교 {np.mean(mr[sel] > mp[sel]):.2f}")
            print("  해석: 두 열이 비슷하면 고장 직전 60초에 정상 변동을 넘는 이탈이 따로 없다는 뜻이고, 고장 직전이 뚜렷이 크면 그만큼이 고장 신호다.")

    if a.placebo_start > 0:
        pe = [e for e in events if e["placebo"] is not None]
        if len(pe) >= 10:
            print(f"\n[5] 센서 간 '관계 이탈' 점수(마할라노비스^2/차원, 정상이면 약 1): 고장 직전 vs 대조 - event {len(pe)}개")
            print(f"{'점수':<34}{'고장 직전 중앙값':>16}{'대조 중앙값':>14}{'대조 95백분위 초과 비율':>24}{'짝 비교(직전이 큰 비율)':>24}")
            for key, nm in (("maha4", "4차원 전체(센서 간 관계)"), ("maha4_major", "  4차원 주축(정상 관계를 따르는 이동)"),
                            ("maha4_minor", "  4차원 나머지 축(관계 이탈)"), ("maha16", "16차원(시간 구조 포함)")):
                r = np.array([e[key] for e in pe])
                pl = np.array([e["placebo"][key] for e in pe])
                p95 = np.percentile(pl, 95)
                print(f"{nm:<34}{np.median(r):>16.2f}{np.median(pl):>14.2f}{np.mean(r > p95):>24.2f}{np.mean(r > pl):>24.2f}")
            print("  해석: 신호가 없다면 두 열이 비슷하고 95백분위 초과 비율 약 0.05, 짝 비교 약 0.50. 고장 직전에 관계가 깨진다면 둘 다 뚜렷이 커진다.")
            print("  (이 점수는 센서 '크기'가 아니라 센서 간 관계를 본다. 앞선 [4]의 max|이동|(크기)과 다른 질문에 답한다.)")

    if a.placebo_start > 0:
        pe = [e for e in events if e["placebo"] is not None]
        if len(pe) >= 10:
            print(f"\n[6] 짝지은 판정(McNemar): 같은 event 의 고장 직전과 대조에서 '이탈 상태'인 비율 - event {len(pe)}개")
            print("    선별 편향을 피하려면 선별된 event 와 아닌 event 를 모두 합친 전체로 보는 것이 맞다 (--mark-csv 로 전체를 분석했을 때).")
            flags = {}
            flags["observable(팀원 기준, 95백분위 초과 >= 50%)"] = ([e["obs_frac"] >= 0.5 for e in pe], [e["placebo"]["obs_frac"] >= 0.5 for e in pe])
            for key, nm in (("maha4", "관계 점수 4차원 전체 > 대조 95백분위"), ("maha4_major", "주축 성분 > 대조 95백분위"),
                            ("maha4_minor", "나머지 축(관계 이탈) > 대조 95백분위")):
                thr_ = np.percentile([e["placebo"][key] for e in pe], 95)
                flags[nm] = ([e[key] > thr_ for e in pe], [e["placebo"][key] > thr_ for e in pe])
            groups = [("전체", np.ones(len(pe), bool))]
            if a.mark_csv:
                tm = np.array([e["in_team"] for e in pe])
                groups += [("팀원이 고른 event", tm), ("그 외 event", ~tm)]
            for gname, gm in groups:
                if gm.sum() < 10:
                    continue
                print(f"\n  [{gname}] event {int(gm.sum())}개")
                print(f"  {'판정':<44}{'직전 비율':>10}{'대조 비율':>10}{'직전만':>8}{'대조만':>8}{'둘다':>6}{'p(McNemar)':>12}")
                for nm, (pre_f, pl_f) in flags.items():
                    pr_, pl_ = np.array(pre_f)[gm], np.array(pl_f)[gm]
                    n10, n01, n11, n00, pv = mcnemar(pr_, pl_)
                    print(f"  {nm:<44}{pr_.mean():>10.2f}{pl_.mean():>10.2f}{n10:>8}{n01:>8}{n11:>6}{pv:>12.3f}")
            print("\n  해석: 직전만 ≈ 대조만 이면 이탈 상태가 고장 직전에 더 잦지 않다(p 가 큼). 직전만 > 대조만 이고 p 가 작으면 고장이 가까워질 때 이탈이 늘어난다.")

    out = [dict(file=e["file"], fault_time=e["fault_time"], fault=e["fault"], cond_key=e["cond_key"], n_pos=e["n_pos"], n_neg=e["n_neg"],
                obs_frac=round(e["obs_frac"], 4), obs_class=e["obs_class"], delta_z=e["delta_z"].round(4).tolist(),
                delta_z_all6=e["delta_z_all6"].round(4).tolist(), dcorr=e["dcorr"].round(4).tolist(),
                maha4=round(e["maha4"], 4), maha4_major=round(e["maha4_major"], 4), maha4_minor=round(e["maha4_minor"], 4), maha16=round(e["maha16"], 4), in_team=bool(e["in_team"]),
                placebo=(None if e["placebo"] is None else dict(obs_frac=round(e["placebo"]["obs_frac"], 4),
                                                                maha4=round(e["placebo"]["maha4"], 4), maha4_major=round(e["placebo"]["maha4_major"], 4), maha4_minor=round(e["placebo"]["maha4_minor"], 4), maha16=round(e["placebo"]["maha16"], 4),
                                                                delta_z=e["placebo"]["delta_z"].round(4).tolist(),
                                                                dcorr=e["placebo"]["dcorr"].round(4).tolist()))) for e in events]
    with open(os.path.join(a.out, "fault_signatures_cache.json"), "w", encoding="utf-8") as f:
        json.dump(dict(sensors_hw=HW, sensors_all=SENSOR_ALL, horizon=a.horizon, excluded_teammate=len(excl), events=out), f, ensure_ascii=False, indent=1)
    print(f"\n저장: {a.out}/fault_signatures_cache.json (event {len(out)}개)")


if __name__ == "__main__":
    main()