#!/usr/bin/env python
"""
[실험] UART 입력 인코딩별 정밀도 손실 측정 (최종 모델 가중치 + Q8.8 골든 모델)

배경: 현재 RTL(input_scaler.sv)은 UART 8비트 값을 {data_in, 8'b0} (0~255 의 부호 없는 정수, Q8.8 의 정수부)로 바꾼다.
      모델의 입력은 표준화된 값(대략 -6 ~ +8, 소수)이라 이 변환으로는 쓸 수 없다. 센서 한 값을 어떻게 UART 로 보낼지 정해야 한다.

비교하는 인코딩 (PC 가 표준화한 값을 정수 q 로 바꿔 보내고, FPGA 가 Q8.8 로 복원)
  Q8.8 16비트        2바이트/값 (현재 학습/검증에 쓴 입력, 반올림)         <- 기준
  Q4.4 8비트 x16     1바이트/값, 범위 -8 ~ +7.94, 분해능 1/16, 반올림     <- 유력 후보
  Q4.4 8비트 x16 절삭 같은 형식인데 PC 가 0 방향으로 절삭 (흔한 구현 실수: int(x*16))
  Q3.5 8비트 x32     범위 -4 ~ +3.97, 분해능 1/32 (범위가 좁아 큰 값이 잘림)
  Q5.3 8비트 x8      범위 -16 ~ +15.9, 분해능 1/8

방법: 평가 세트(표준화된 float 입력)를 인코딩 -> Q8.8 복원 -> Q8.8 골든 모델(RTL 수정 4가지 적용) 로 점수/판정을 만들고
      (1) FP32 와의 판정 일치, (2) 16비트 기준과의 판정 일치, (3) AUC, (4) 임계값 두 개에서의 TPR/FPR 을 본다.

사전 판정 기준 (결과를 보기 전에 정해 둔 것)
  8비트 인코딩을 채택하는 조건: 16비트 대비 AUC 하락 0.005 이하 이고 16비트와의 판정 일치율 0.99 이상.
  조건을 만족하는 8비트 인코딩 중 아래 순서로 고른다: Q4.4 반올림 -> Q3.5 -> Q5.3. 하나도 없으면 16비트 인코딩.

사용법
  python exp_input_encoding.py --weights res_square_out/model_fp32.npz --data dataset_t_hard --mem-dir ../mem
  python exp_input_encoding.py --weights ... --data dataset_corr --type-names corr_break --mem-dir ../mem

필요: q88_golden.py 를 같은 폴더에 둔다. numpy, scikit-learn.
"""
import argparse
import json
import os

import numpy as np

import q88_golden as g

ENCODINGS = [
    # 이름, 비트, 배율, 모드
    ("Q8.8 16비트 (반올림, 기준)", 16, 256, "round"),
    ("Q4.4 8비트 x16 (반올림)", 8, 16, "round"),
    ("Q4.4 8비트 x16 (0 방향 절삭)", 8, 16, "trunc"),
    ("Q3.5 8비트 x32 (범위 +-4)", 8, 32, "round"),
    ("Q5.3 8비트 x8 (범위 +-16)", 8, 8, "round"),
]


def encode_to_q88(X, bits, scale, mode):
    """float 입력 -> (정수 q 로 양자화/포화) -> Q8.8 원시값(int64). 반환: (Xq88, 포화된 값의 비율)"""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    v = X.astype(np.float64) * scale
    q = np.rint(v) if mode == "round" else np.trunc(v)
    qc = np.clip(q, lo, hi)
    sat = float(np.mean(q != qc))
    return (qc * (256 // scale)).astype(np.int64), sat


def auc_safe(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def load_weights(path):
    z = np.load(path)
    w = {k: z[k].astype(np.float64) for k in ("wq", "wk", "wv", "wfc")}
    w["arch"] = np.array(str(z["arch"]) if "arch" in z.files else "rtl")
    for k in ("b", "c"):
        if k in z.files:
            w[k] = z[k].astype(np.float64)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="train_attention.py 가 만든 model_fp32.npz")
    ap.add_argument("--data", required=True, help="평가 데이터셋 폴더 (dataset_test.npz 를 읽음)")
    ap.add_argument("--mem-dir", required=True, help="저장소 mem 폴더 (softmax ROM)")
    ap.add_argument("--type-names", default="offset,ramp,swap_cond", help="이상 유형 이름(atype 0,1,2 순서)")
    ap.add_argument("--thr", type=float, default=0.5, help="판정 임계값 1 (RTL 현재값 0.5)")
    ap.add_argument("--thr2", type=float, default=-0.131, help="판정 임계값 2 (Val 에서 TPR-FPR 이 최대였던 값)")
    ap.add_argument("--out", default="enc_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    w = load_weights(args.weights)
    arch = str(w["arch"])
    if arch not in ("res_square", "square", "rtl"):
        raise SystemExit(f"골든 모델이 지원하지 않는 구조입니다: {arch}")
    te = np.load(os.path.join(args.data, "dataset_test.npz"))
    X, y = te["X"].astype(np.float64), te["y"]
    at = te["atype"] if "atype" in te.files else None
    names = [s.strip() for s in args.type_names.split(",")]
    print(f"가중치 {args.weights} (arch={arch}), 평가 {args.data}: {len(y):,}개 (이상 {int(y.sum()):,}), 입력 범위 {X.min():.2f} ~ {X.max():.2f}")

    wi = {k: g.to_q88(v) for k, v in w.items() if k in g.NUM_KEYS}
    roms = g.make_roms("repo", args.mem_dir)
    kw = dict(mul_bits=32, fc_signed=True, slice_start=3, inv_mode="full", arch=arch,
              b_int=int(wi["b"]) if "b" in wi else 0, c_int=wi.get("c"))
    wargs = (wi["wq"], wi["wk"], wi["wv"], wi["wfc"], roms)
    z32 = g.fwd_fp32_w(w, X)[0]

    def metrics(score_f, ref_label32, base_label=None):
        out = dict(auc=auc_safe(y, score_f))
        if at is not None:
            for t, nm in enumerate(names):
                m = (y == 0) | (at == t)
                out[nm] = auc_safe(y[m], score_f[m]) if (at == t).sum() > 20 else float("nan")
        for tag, thr in (("a", args.thr), ("b", args.thr2)):
            lab = score_f > thr
            out[f"tpr_{tag}"] = float(lab[y == 1].mean())
            out[f"fpr_{tag}"] = float(lab[y == 0].mean())
        lab = score_f > args.thr
        out["agree_fp32"] = float((lab == ref_label32).mean())
        out["agree_base"] = float((lab == base_label).mean()) if base_label is not None else float("nan")
        return out, lab

    ref32 = z32 > args.thr
    rows = []
    r, _ = metrics(z32, ref32)
    rows.append(dict(name="FP32 (참고, 양자화 없음)", bytes_per_value="-", rng="-", step="-", sat=0.0, score_mae=0.0, score_max=0.0, **r))
    base_label = None
    for name, bits, scale, mode in ENCODINGS:
        xq, sat = encode_to_q88(X, bits, scale, mode)
        sc = g.forward_q88(xq, *wargs, **kw)["score"] / 256.0
        r, lab = metrics(sc, ref32, base_label)
        if base_label is None:
            base_label = lab
            r["agree_base"] = 1.0
        lo, hi = -(1 << (bits - 1)) / scale, ((1 << (bits - 1)) - 1) / scale
        rows.append(dict(name=name, bytes_per_value=bits // 8, rng=f"{lo:g}~{hi:.2f}", step=f"1/{scale}" if scale < 256 else "1/256",
                         sat=sat, score_mae=float(np.abs(sc - z32).mean()), score_max=float(np.abs(sc - z32).max()), **r))

    print("\n=== 입력 인코딩별 결과 (Q8.8 골든 모델, RTL 수정 4가지 적용) ===")
    head = f"{'인코딩':<30}{'바이트':>6}{'범위':>14}{'분해능':>8}{'포화':>8}{'FP32일치':>9}{'16비트일치':>11}{'점수오차 평균/최대':>20}{'AUC':>8}"
    for nm in names:
        head += f"{nm[:8]:>9}" if at is not None else ""
    head += f"{'TPR/FPR@%.2f' % args.thr:>16}{'TPR/FPR@%.3f' % args.thr2:>17}"
    print(head)
    for r in rows:
        line = (f"{r['name']:<30}{str(r['bytes_per_value']):>6}{r['rng']:>14}{r['step']:>8}{r['sat']:>8.4f}{r['agree_fp32']:>9.3f}"
                f"{r['agree_base']:>11.3f}{r['score_mae']:>10.4f}/{r['score_max']:<9.3f}{r['auc']:>8.3f}")
        if at is not None:
            for nm in names:
                line += f"{r[nm]:>9.3f}"
        line += f"{r['tpr_a']:>9.3f}/{r['fpr_a']:<6.3f}{r['tpr_b']:>9.3f}/{r['fpr_b']:<6.3f}"
        print(line)

    base = next(r for r in rows if r["name"].startswith("Q8.8"))
    print("\n=== 사전 판정 기준 적용 (16비트 대비 AUC 하락 <= 0.005 이고 판정 일치 >= 0.99) ===")
    chosen = None
    for r in rows:
        if r["name"].startswith(("FP32", "Q8.8")):
            continue
        drop = base["auc"] - r["auc"]
        ok = drop <= 0.005 and r["agree_base"] >= 0.99
        print(f"  {r['name']:<30} AUC 하락 {drop:+.4f}, 16비트와 판정 일치 {r['agree_base']:.3f} -> {'충족' if ok else '미충족'}")
    for nm in ("Q4.4 8비트 x16 (반올림)", "Q3.5 8비트 x32 (범위 +-4)", "Q5.3 8비트 x8 (범위 +-16)"):
        r = next(r for r in rows if r["name"] == nm)
        if base["auc"] - r["auc"] <= 0.005 and r["agree_base"] >= 0.99:
            chosen = nm
            break
    print(f"\n  -> 선택: {chosen if chosen else 'Q8.8 16비트 (8비트 인코딩이 기준을 만족하지 못함)'}")
    print("\n[구현 메모 - 선택이 Q4.4 8비트 x16 일 때]")
    print("  PC 측 : q = np.clip(np.rint(x_std * 16), -128, 127).astype(np.int8); byte = int(q) & 0xFF   (x_std = 센서별 (값-mean)/std)")
    print("  RTL   : input_scaler.sv 의 scaled_out <= {data_in, 8'b0}  ->  {{4{data_in[7]}}, data_in, 4'b0000}  (부호 확장 후 4비트 왼쪽 시프트 = Q8.8)")
    print("  절삭(int(x*16))으로 보내면 0 근처 값이 한쪽으로 치우쳐 편향이 생긴다. 반드시 반올림을 쓴다.")
    with open(os.path.join(args.out, "enc_results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), rows=rows, chosen=chosen), f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장: {args.out}/enc_results.json")


if __name__ == "__main__":
    main()