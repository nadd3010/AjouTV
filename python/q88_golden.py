#!/usr/bin/env python
"""
Q8.8 골든 모델 (NumPy). RTL(rtl/attention_engine, rtl/classifier)의 정수 연산을 그대로 재현한다.

RTL 파이프라인 (입력 x = 센서 4행 x 시간 4열, Q8.8)
  qkv_transform : q = slice(x @ Wq), k = slice(x @ Wk), v = slice(x @ Wv)      slice = 누산기 [23:8]
  score_compute : score = slice((q @ k^T) >>> 1)                                 (sqrt(dk)=2 를 시프트로)
  softmax_lut   : 행별 max 빼기 -> exp_rom[255 - (|x-max| >> 4)] -> 합 -> inv_rom[합 인덱스]
                  attn_weight = (exp * inv) >> 8
  attn_output   : attn_out = slice(attn_weight @ v)
  fc_layer      : score_abnormal = slice(sum_i attn_out[i] * w_fc[i])           (바이어스 없음)
  argmax        : label = (score_abnormal > 0x0080)                              (임시 threshold 0.5)
  cause_identifier : col_sum[j] = sum_i attn_weight[i][j], cause_index = argmax_j col_sum[j]

저장소(github.com/nadd3010/AjouTV) main 의 rtl/ 과 mem/ 을 읽고 확인한 불일치 4가지.
모두 이 모델에서 옵션으로 켜고 끌 수 있고, FP32 와의 차이가 얼마나 줄어드는지 표로 확인한다.
  (1) fc_layer: w_fc 가 unsigned([15:0]) 라서 `$signed(attn_out) * w_fc[idx]` 가 unsigned 곱이 된다.
      attn_out = 1.0, 2.0 (256의 배수)에서는 오차가 [23:8] 밖에 있어 tb_fc_layer 가 통과하지만 일반 값에서는 틀린다.  -> fc_signed
  (2) softmax_lut: `(exp_buffer * current_inv_val) >> 8` 이 16비트 문맥에서 계산되어 곱이 65536 이상이면 >>8 전에 잘린다.
      (mem/softmax_inv_rom.mem 의 값은 최대 0xFFFF 이므로 실제로 걸린다.)                                           -> mul_bits
  (3) softmax_lut: inv_rom 이 [0:255] 이고 인덱스가 exp_sum[15:8](합의 정수부, 0~4) 인데,
      mem/softmax_inv_rom.mem 은 1025칸이며 '합의 Q8.8 원시값(0~1024)'으로 인덱싱하도록 만들어져 있다
      (make_inverse_rom.py: inv[i] = 65536 / i).                                                               -> inv_mode
  (4) softmax_lut: exp 인덱스는 |x-max| >> SLICE_START(=4) 라서 한 칸이 1/16 이지만,
      mem/softmax_exp_rom.mem 은 [-8, 0] 을 256칸으로 나눈 것(make_lut.py)이라 한 칸이 8/255 = 약 1/32 이다.
      그대로 쓰면 exp(-|x|/2) 가 되어 softmax 가 2배 평평해진다.                                                 -> slice_start (3 이면 맞음)

사용법
  python q88_golden.py --selftest
  (학습 후 비교는 train_attention.py 가 이 파일의 함수를 호출한다)

필요 패키지: numpy (비교표의 AUC 계산에만 scikit-learn)
"""
import argparse
import os

import numpy as np

THRESHOLD_Q88 = 0x0080      # argmax.sv 의 임시 threshold (0.5)


# ---------------------------------------------------------------------------
# 기본 변환
# ---------------------------------------------------------------------------
def to_q88(x):
    """float -> Q8.8 정수(int64), 반올림 + int16 포화."""
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * 256.0), -32768, 32767).astype(np.int64)


def from_q88(i):
    return np.asarray(i, dtype=np.float64) / 256.0


def wrap16(v):
    """하위 16비트만 취해 signed 로 해석 (RTL 비트 슬라이스와 동일)."""
    v = np.asarray(v, dtype=np.int64) & 0xFFFF
    return np.where(v >= 0x8000, v - 0x10000, v)


def slice_23_8(acc, stats=None, key=None):
    """누산기 [23:8] 슬라이스 = (acc >> 8) 의 하위 16비트. 포화가 아니라 wrap 이다."""
    ideal = acc >> 8
    out = wrap16(ideal)
    if stats is not None and key is not None:
        stats[key] = max(stats.get(key, 0.0), float(np.mean(out != ideal)))
    return out


def hexw(v):
    return f"{int(v) & 0xFFFF:04X}"


def load_mem(path):
    """$readmemh 와 같은 방식(공백/빈 줄/주석 무시)으로 16진수 워드를 읽는다."""
    vals = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.split("//")[0].strip()
            for tok in line.split():
                vals.append(int(tok, 16))
    return np.array(vals, dtype=np.int64)


# ---------------------------------------------------------------------------
# Softmax ROM
# ---------------------------------------------------------------------------
class Roms:
    def __init__(self, exp_rom, inv_rom, name):
        self.exp_rom = np.asarray(exp_rom, dtype=np.int64)       # 256칸
        self.inv_rom = np.asarray(inv_rom, dtype=np.int64)       # 파일 전체 (1025칸)
        self.name = name


def make_roms(kind="repo", mem_dir=None):
    """kind='repo' : 저장소 mem 파일을 그대로 읽음 (mem_dir 필요).
       kind='generated' : make_lut.py([-8,0] 256칸) 와 make_inverse_rom.py(1025칸) 공식으로 같은 값을 생성."""
    if kind == "repo" and mem_dir and os.path.exists(os.path.join(mem_dir, "softmax_exp_rom.mem")):
        e = load_mem(os.path.join(mem_dir, "softmax_exp_rom.mem"))
        v = load_mem(os.path.join(mem_dir, "softmax_inv_rom.mem"))
        return Roms(e, v, "repo")
    e = np.round(256.0 * np.exp(np.linspace(-8.0, 0.0, 256))).astype(np.int64)
    v = np.zeros(1025, dtype=np.int64)
    v[1:] = np.minimum(np.round(65536.0 / np.arange(1, 1025)), 65535).astype(np.int64)
    return Roms(e, v, "generated")


# ---------------------------------------------------------------------------
# 단계별 연산 (RTL 순서)
# ---------------------------------------------------------------------------
def qkv(x, wq, wk, wv, stats=None):
    f = lambda w, k: slice_23_8(np.einsum("brd,dc->brc", x, w), stats, k)
    return f(wq, "qkv_wrap"), f(wk, "qkv_wrap"), f(wv, "qkv_wrap")


def scores(q, k, stats=None):
    dot = np.einsum("bid,bjd->bij", q, k)          # q @ k^T
    return slice_23_8(dot >> 1, stats, "score_wrap")


def softmax_q88(score, roms, mul_bits=16, slice_start=4, inv_mode="rtl"):
    """softmax_lut.sv 재현.
       slice_start : exp 인덱스 = 255 - (|x-max| >> slice_start). RTL 은 4.
       inv_mode    : 'rtl'  = inv_rom[0:255], 인덱스 exp_sum[15:8] (RTL 그대로)
                     'full' = inv_rom[0:1024], 인덱스 exp_sum (mem 파일 규약, RTL 수정 필요)
       mul_bits    : 16 이면 곱을 16비트로 자른 뒤 >>8 (RTL 그대로), 32 이면 자르지 않음."""
    mx = score.max(axis=2, keepdims=True)
    shifted = wrap16(score - mx)
    abs16 = (-shifted) & 0xFFFF                            # RTL: $unsigned(-shifted_score), 16비트
    idx = np.where(shifted < -(255 << slice_start), 0, (255 - (abs16 >> slice_start)) & 0xFF)   # 8비트 인덱스
    ex = roms.exp_rom[idx]
    s = ex.sum(axis=2)                                     # (B,4) 18비트 합
    if inv_mode == "rtl":
        inv = roms.inv_rom[:256][(s >> 8) & 0xFF]
    else:
        inv = roms.inv_rom[np.minimum(s, len(roms.inv_rom) - 1)]
    prod = ex * inv[:, :, None]
    if mul_bits == 16:
        prod = prod & 0xFFFF
    return wrap16(prod >> 8)


def attn_output(aw, v, stats=None):
    return slice_23_8(np.einsum("bik,bkj->bij", aw, v), stats, "attn_out_wrap")


def fc_q88(ao, wfc, fc_signed, stats=None, arch="rtl", b_int=0, c_int=None, xq=None):
    flat = ao.reshape(len(ao), 16)
    if arch == "res_square":                       # Q8.8 덧셈 (16비트 wrap)
        flat = wrap16(flat + np.asarray(xq, dtype=np.int64).reshape(len(ao), 16))
    if arch != "rtl":                              # 구조 변경안: 부호를 처음부터 올바르게 설계한다고 가정
        x = flat if arch in ("square", "res_square") else wrap16(flat - c_int[None, :])
        sq = slice_23_8(x * x, stats, "sq_wrap")   # (x*x) >> 8 의 하위 16비트 (Q8.8)
        out = slice_23_8(sq @ wfc, stats, "fc_wrap")
        return wrap16(out + int(b_int))
    if fc_signed:
        acc = flat @ wfc
    else:                                          # RTL: w_fc 가 unsigned -> 곱 전체가 unsigned (36비트)
        acc = ((flat & 0xFFFF) * (wfc[None, :] & 0xFFFF)).sum(axis=1) & ((1 << 36) - 1)
    return slice_23_8(acc, stats, "fc_wrap")


def forward_q88(xq, wq, wk, wv, wfc, roms, mul_bits=16, fc_signed=False, slice_start=4, inv_mode="rtl",
                arch="rtl", b_int=0, c_int=None):
    """xq: (B,4,4) Q8.8 정수, w*: Q8.8 정수 가중치. 기본값은 main RTL 그대로이다."""
    stats = {}
    x = np.asarray(xq, dtype=np.int64)
    q, k, v = qkv(x, wq, wk, wv, stats)
    sc = scores(q, k, stats)
    aw = softmax_q88(sc, roms, mul_bits, slice_start, inv_mode)
    ao = attn_output(aw, v, stats)
    sa = fc_q88(ao, wfc, fc_signed, stats, arch, b_int, c_int, xq=x)
    label = (sa > THRESHOLD_Q88).astype(np.int64)
    colsum = aw.sum(axis=1)                        # 열 합산 (행 방향 합)
    cause = colsum.argmax(axis=1)                  # 동률이면 앞 인덱스 (RTL 의 strict >)
    return dict(score=sa, label=label, cause=cause, colsum=colsum, aw=aw, stats=stats)


NUM_KEYS = ("wq", "wk", "wv", "wfc", "b", "c")


def forward_fp32(x, wq, wk, wv, wfc, arch="rtl", b=0.0, c=None):
    """같은 구조의 부동소수점 기준값(NumPy). 학습 모델과 같은 식이다.
       arch='rtl'    : z = flatten(O) . w_fc                      (현재 RTL)
       arch='square' : z = flatten(O)^2 . w_fc + b                (FC 앞에 제곱 한 단계)
       arch='res_square' : z = flatten(O + X)^2 . w_fc + b        (잔차 연결 + 제곱)
       arch='dist'   : z = (flatten(O) - c)^2 . w_fc + b          (정상 중심 c 와의 거리)"""
    q, k, v = x @ wq, x @ wk, x @ wv
    s = np.einsum("bid,bjd->bij", q, k) / 2.0
    e = np.exp(s - s.max(axis=2, keepdims=True))
    a = e / e.sum(axis=2, keepdims=True)
    o = np.einsum("bik,bkj->bij", a, v)
    if arch == "res_square":
        o = o + x                                  # 잔차 연결: attention 출력에 원래 입력(센서 x 시간)을 더함
    f = o.reshape(len(x), 16)
    if arch == "rtl":
        z, sq = f @ wfc, None
    elif arch in ("square", "res_square"):
        sq = f ** 2
        z = sq @ wfc + b
    elif arch == "dist":
        sq = (f - c[None, :]) ** 2
        z = sq @ wfc + b
    else:
        raise ValueError(arch)
    rng = dict(q=np.abs(q).max(), k=np.abs(k).max(), v=np.abs(v).max(), score=np.abs(s).max(),
               attn_out=np.abs(o).max(), z=np.abs(z).max())
    if sq is not None:
        rng["sq"] = np.abs(sq).max()
    return z, a, rng


def fwd_fp32_w(w, x):
    """가중치 dict(w) 로 forward_fp32 호출. arch/b/c 는 있으면 사용."""
    arch = str(w.get("arch", "rtl"))
    return forward_fp32(x, w["wq"], w["wk"], w["wv"], w["wfc"], arch,
                        float(w.get("b", 0.0)), w.get("c"))


# ---------------------------------------------------------------------------
# FP32 vs Q8.8 비교 (RTL 수정 후보를 단계별로 켜 보는 표)
# ---------------------------------------------------------------------------
def compare_fp32_q88(w, X, Xq, y, cause, mem_dir=None, thr=0.5):
    """w = dict(wq,wk,wv: (4,4) float, wfc: (16,) float). X float, Xq Q8.8 정수 입력."""
    from sklearn.metrics import roc_auc_score
    arch = str(w.get("arch", "rtl"))
    z32, a32, rng = fwd_fp32_w(w, X)
    lab32 = (z32 > thr).astype(np.int64)
    cause32 = a32.sum(axis=1).argmax(axis=1)
    anom = y == 1
    wi = {k: to_q88(v) for k, v in w.items() if k in NUM_KEYS}
    kwq = dict(arch=arch, b_int=int(wi["b"]) if "b" in wi else 0, c_int=wi.get("c"))

    def row(name, lab, zz, cs, extra=""):
        return dict(name=name, acc=float((lab == y).mean()), auc=float(roc_auc_score(y, zz)),
                    agree=float((lab == lab32).mean()), z_mae=float(np.abs(zz - z32).mean()),
                    z_max=float(np.abs(zz - z32).max()),
                    cause_acc=float((cs[anom] == cause[anom]).mean()),
                    cause_agree=float((cs == cause32).mean()), note=extra)

    rows = [row("FP32 (기준)", lab32, z32, cause32)]
    roms = make_roms("repo", mem_dir)
    src = "저장소 mem" if roms.name == "repo" else "생성 ROM"
    variants = [
        (f"main RTL 그대로 ({src})", dict(mul_bits=16, fc_signed=False, slice_start=4, inv_mode="rtl")),
        ("+ (1) fc_layer signed", dict(mul_bits=16, fc_signed=True, slice_start=4, inv_mode="rtl")),
        ("+ (2) softmax 곱 32비트", dict(mul_bits=32, fc_signed=True, slice_start=4, inv_mode="rtl")),
        ("+ (3) inv ROM 1025칸/인덱스 sum", dict(mul_bits=32, fc_signed=True, slice_start=4, inv_mode="full")),
        ("+ (4) exp SLICE_START=3", dict(mul_bits=32, fc_signed=True, slice_start=3, inv_mode="full")),
    ]
    for name, kw in variants:
        out = forward_q88(Xq, wi["wq"], wi["wk"], wi["wv"], wi["wfc"], roms, **kw, **kwq)
        wraps = max(out["stats"].values()) if out["stats"] else 0.0
        rows.append(row(name, out["label"], out["score"] / 256.0, out["cause"], f"wrap 최대 {wraps:.4f}"))
    return rows, rng


def _zero_row(x, r):
    x2 = np.array(x, copy=True)
    x2[:, r, :] = 0
    return x2


def loo_compare_q88(w, X, Xq, y, cause, mem_dir=None):
    """Leave-one-sensor-out 원인 센서: 센서 r 의 행을 0(Train 평균)으로 바꿨을 때 이상 점수가 가장 많이 줄어드는 센서.
    (교수님 5번의 perturbation 방식. 하드웨어에서는 파이프라인을 센서마다 한 번씩 더 돌리면 된다.)
    FP32 와 Q8.8(RTL 수정 4가지 적용) 결과를 이상 윈도우에서 비교한다."""
    anom = y == 1
    Xa, Xqa, ca = X[anom], Xq[anom], cause[anom]
    z0 = fwd_fp32_w(w, Xa)[0]
    d32 = np.stack([z0 - fwd_fp32_w(w, _zero_row(Xa, r))[0] for r in range(4)], axis=1)
    arch = str(w.get("arch", "rtl"))
    wi = {k: to_q88(v) for k, v in w.items() if k in NUM_KEYS}
    roms = make_roms("repo", mem_dir)
    kw = dict(mul_bits=32, fc_signed=True, slice_start=3, inv_mode="full",
              arch=arch, b_int=int(wi["b"]) if "b" in wi else 0, c_int=wi.get("c"))
    args = (wi["wq"], wi["wk"], wi["wv"], wi["wfc"], roms)
    s0 = forward_q88(Xqa, *args, **kw)["score"]
    d88 = np.stack([(s0 - forward_q88(_zero_row(Xqa, r), *args, **kw)["score"]) / 256.0 for r in range(4)], axis=1)
    p32, p88 = d32.argmax(axis=1), d88.argmax(axis=1)
    m32 = np.sort(d32, axis=1)
    m88 = np.sort(d88, axis=1)
    return dict(n=int(len(ca)), acc32=float((p32 == ca).mean()), acc88=float((p88 == ca).mean()),
                agree=float((p32 == p88).mean()),
                margin32=float(np.median(m32[:, -1] - m32[:, -2])), margin88=float(np.median(m88[:, -1] - m88[:, -2])))


def print_rows(rows, rng=None):
    print(f"{'구성':<34}{'Acc':>7}{'AUC':>7}{'FP32와 라벨일치':>14}{'z 평균오차':>11}{'z 최대오차':>11}"
          f"{'원인정확도':>10}{'원인일치':>9}  비고")
    for r in rows:
        print(f"{r['name']:<34}{r['acc']:>7.3f}{r['auc']:>7.3f}{r['agree']:>14.3f}{r['z_mae']:>11.3f}"
              f"{r['z_max']:>11.2f}{r['cause_acc']:>10.3f}{r['cause_agree']:>9.3f}  {r['note']}")
    if rng:
        print("\nFP32 중간값 최대 |값| (Q8.8 표현 한계는 128, 넘으면 wrap 위험):")
        print("  " + ", ".join(f"{k}={v:.1f}" for k, v in rng.items()))


# ---------------------------------------------------------------------------
# .mem 내보내기
# ---------------------------------------------------------------------------
def export_mem(out_dir, w):
    """weights.mem(48워드) + fc_weights.mem(16워드). arch 가 square/dist 면 fc_extra.mem 도 쓴다:
       square -> [b] (1워드),  dist -> [c0..c15, b] (17워드)."""
    os.makedirs(out_dir, exist_ok=True)
    wi = {k: to_q88(v) for k, v in w.items() if k in NUM_KEYS}
    with open(os.path.join(out_dir, "weights.mem"), "w") as f:
        for g, key in enumerate(("wq", "wk", "wv")):
            for i in range(4):
                for j in range(4):
                    f.write(hexw(wi[key][i, j]) + "\n")      # 인덱스 = g*16 + i*4 + j (qkv_transform.sv 와 동일)
            if g < 2:
                f.write("\n")
    with open(os.path.join(out_dir, "fc_weights.mem"), "w") as f:
        for v in wi["wfc"]:
            f.write(hexw(v) + "\n")
    arch = str(w.get("arch", "rtl"))
    if arch != "rtl":
        with open(os.path.join(out_dir, "fc_extra.mem"), "w") as f:
            if arch == "dist":
                for v in np.atleast_1d(wi["c"]):
                    f.write(hexw(v) + "\n")
            f.write(hexw(int(wi["b"])) + "\n")


# ---------------------------------------------------------------------------
# 자체 점검: RTL 테스트벤치 값과 같은지, 의심 지점이 실제로 차이를 만드는지
# ---------------------------------------------------------------------------
def selftest():
    fc_w = np.array([0xFF1C, 0x00A4, 0x0032, 0x002F, 0xFFCA, 0xFF69, 0x0035, 0xFF49,
                     0x0029, 0xFFE5, 0x0052, 0x004B, 0x0127, 0x0013, 0xFF76, 0x0044], dtype=np.int64)
    fc_s = wrap16(fc_w)
    print("[1] fc_layer: tb_fc_layer.sv 와 같은 가중치 (fc_weights.mem)")
    for val, exp in ((0x0100, 0x0071), (0x0200, 0x00E2)):
        ao = np.full((1, 4, 4), val, dtype=np.int64)
        r_unsigned = int(fc_q88(ao, fc_w, False)[0]) & 0xFFFF
        r_signed = int(fc_q88(ao, fc_s, True)[0]) & 0xFFFF
        print(f"    입력 {val:#06x}: RTL식(unsigned)={r_unsigned:#06x}, signed={r_signed:#06x}, TB 기대={exp:#06x} "
              f"-> {'일치' if r_unsigned == exp and r_signed == exp else '불일치'}")
    rs = np.random.default_rng(0)
    ao = rs.integers(-300, 300, size=(2000, 4, 4))
    a = fc_q88(ao, fc_w, False)
    b = fc_q88(ao, fc_s, True)
    print(f"    무작위 입력 2000개: unsigned 방식과 signed 방식의 결과가 다른 비율 = {float(np.mean(a != b)):.3f}")
    ao = np.full((1, 4, 4), 0x0040, dtype=np.int64)
    print(f"    입력 0x0040(0.25): unsigned={int(fc_q88(ao, fc_w, False)[0]) & 0xFFFF:#06x}, "
          f"signed={int(fc_q88(ao, fc_s, True)[0]) & 0xFFFF:#06x}  <- ModelSim 에서 이 값으로 TB 를 돌려 확인")

    print("[2] softmax_lut: 한 원소만 크고 나머지는 매우 작은 행 (기대: 첫 원소 0x0100 = 1.0, 나머지 0)")
    sc = np.array([[[0, -2000, -2000, -2000]] * 4], dtype=np.int64)
    roms = make_roms("generated")
    hx = lambda w: [hex(int(v) & 0xFFFF) for v in w[0, 0]]
    print(f"    main RTL 그대로 (곱 16비트, inv 정수부 인덱스): {hx(softmax_q88(sc, roms, 16, 4, 'rtl'))}")
    print(f"    + 곱 32비트                                  : {hx(softmax_q88(sc, roms, 32, 4, 'rtl'))}")
    print(f"    + inv ROM 1025칸/인덱스 sum                  : {hx(softmax_q88(sc, roms, 32, 4, 'full'))}")
    sc0 = np.zeros((1, 4, 4), dtype=np.int64)
    print(f"    점수가 모두 같을 때(기대 0x0040 = 0.25), 수정 모두 적용: "
          f"{hx(softmax_q88(sc0, roms, 32, 3, 'full'))}")

    print("[3] exp 인덱스 간격: 1칸이 실제 입력으로 얼마인가 (ROM 은 [-8,0] 을 256칸 = 8/255)")
    for sl in (4, 3):
        d = np.array([[[0, -256, -512, -1024]] * 4], dtype=np.int64)      # 0, -1, -2, -4 (Q8.8)
        w = softmax_q88(d, roms, 32, sl, "full")[0, 0] / 256.0
        ref = np.exp([0, -1, -2, -4]); ref = ref / ref.sum()
        print(f"    SLICE_START={sl}: attn_weight={np.round(w, 3).tolist()}  이상적인 softmax={np.round(ref, 3).tolist()}")

    print("[3b] 수정을 모두 적용한 정수 파이프라인이 float 과 어긋나는 정도 (작은 값)")
    rs = np.random.default_rng(1)
    w = dict(wq=rs.normal(0, 0.4, (4, 4)), wk=rs.normal(0, 0.4, (4, 4)), wv=rs.normal(0, 0.4, (4, 4)),
             wfc=rs.normal(0, 0.2, 16))
    X = rs.normal(0, 0.6, (500, 4, 4))
    z32, a32, _ = forward_fp32(X, w["wq"], w["wk"], w["wv"], w["wfc"])
    wi = {k: to_q88(v) for k, v in w.items()}
    out = forward_q88(to_q88(X), wi["wq"], wi["wk"], wi["wv"], wi["wfc"], roms, 32, True, 3, "full")
    print(f"    |z_q88 - z_fp32| 평균 = {float(np.abs(out['score'] / 256.0 - z32).mean()):.4f}, "
          f"최대 = {float(np.abs(out['score'] / 256.0 - z32).max()):.4f}")
    print(f"    attn_weight 행합 (이상적으로 1.0): {float((out['aw'].sum(axis=2) / 256.0).mean()):.3f}")

    print("[4] .mem 내보내기 / 읽기 왕복")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        export_mem(d, w)
        m = load_mem(os.path.join(d, "weights.mem"))
        f2 = load_mem(os.path.join(d, "fc_weights.mem"))
        ok = (len(m) == 48 and len(f2) == 16 and
              np.array_equal(wrap16(m[:16]).reshape(4, 4), wi["wq"]) and
              np.array_equal(wrap16(m[32:]).reshape(4, 4), wi["wv"]) and np.array_equal(wrap16(f2), wi["wfc"]))
        print(f"    weights.mem {len(m)}워드, fc_weights.mem {len(f2)}워드, 값 왕복 일치 = {ok}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        print("python q88_golden.py --selftest")