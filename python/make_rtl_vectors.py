#!/usr/bin/env python
"""
RTL 검증 키트 생성기: 골든 모델(q88_golden.py)로 가중치 .mem 파일과 단계별 기대값 벡터를 만든다.

  python make_rtl_vectors.py --mode fixed --weights res_square_out/model_fp32.npz \
        --data dataset_t_hard --mem-dir ../mem --out kit_fixed
  python make_rtl_vectors.py --mode orig  --mem-dir ../mem --out kit_orig      # 지금 main RTL 의 동작 확인용

  --mode fixed : 수정한 RTL(res_square + 버그 4가지 수정)용. 가중치는 --weights(model_fp32.npz) 에서 만든다.
                 (--demo 를 주면 무작위 가중치로 배선 점검용 키트를 만든다)
  --mode orig  : 현재 main RTL 용. 가중치는 저장소 mem 의 weights.mem / fc_weights.mem 을 그대로 쓴다.

출력 폴더(out)
  *.mem 가중치/ROM 파일 (weights.mem, fc_weights.mem, fc_extra.mem, softmax_exp_rom.mem, softmax_inv_rom.mem)
  vec_n.mem          윈도우 개수
  vec_x.mem          입력 윈도우 (윈도우당 16워드, 행 우선 = 센서 x 시간, Q8.8 16진수)
  exp_q/k/v/score/aw/ao.mem   단계별 기대값 (윈도우당 16워드)
  exp_z.mem, exp_label.mem, exp_cause.mem   윈도우당 1워드 (z = score_abnormal)

윈도우 구성: 데이터셋 Test 의 윈도우(--data 가 있을 때) + 경계값/포화/무작위 윈도우.
"""
import argparse
import os
import shutil
import sys

import numpy as np

# q88_golden.py 를 같은 폴더, 상위 폴더, 현재 폴더 어디에 두어도 찾는다
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here), os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from q88_golden import (THRESHOLD_Q88, Roms, attn_output, export_mem, fc_q88, load_mem, make_roms, qkv, scores,
                        softmax_q88, to_q88, wrap16)


def hexw(v):
    return f"{int(v) & 0xFFFF:04X}"


def write_words(path, arr):
    with open(path, "w") as f:
        for v in np.asarray(arr).reshape(-1):
            f.write(hexw(v) + "\n")


def edge_windows(rs, n_random):
    ws = [np.zeros((4, 4)), np.full((4, 4), 0x7FFF), np.full((4, 4), -0x8000),
          np.full((4, 4), 256), np.full((4, 4), -256)]
    alt = np.zeros((4, 4))
    alt[::2, :] = 0x0200
    alt[1::2, :] = -0x0200
    ws.append(alt)
    for _ in range(n_random):                       # 일반 범위 (표준화 입력 +-3 정도)
        ws.append(rs.integers(-768, 768, size=(4, 4)))
    for _ in range(n_random // 2):                  # 넓은 범위 (wrap/포화 경로까지 비트 단위로 확인)
        ws.append(rs.integers(-0x4000, 0x4000, size=(4, 4)))
    return np.array(ws, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed", "orig"], required=True)
    ap.add_argument("--weights", default=None, help="train_attention.py 가 만든 model_fp32.npz (fixed 모드)")
    ap.add_argument("--demo", action="store_true", help="무작위 가중치로 키트를 만든다 (배선 점검용)")
    ap.add_argument("--arch", default="res_square", choices=["res_square", "square", "rtl"], help="--demo 일 때 구조")
    ap.add_argument("--data", default=None, help="dataset 폴더 (Test 윈도우를 입력으로 사용)")
    ap.add_argument("--mem-dir", required=True, help="저장소 mem 폴더 (softmax ROM 및 orig 모드 가중치)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-data", type=int, default=120, help="데이터셋에서 가져올 윈도우 수")
    ap.add_argument("--n-random", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rs = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ---------------- 가중치 ----------------
    if args.mode == "orig":
        m = wrap16(load_mem(os.path.join(args.mem_dir, "weights.mem")))
        f = wrap16(load_mem(os.path.join(args.mem_dir, "fc_weights.mem")))
        wi = dict(wq=m[:16].reshape(4, 4), wk=m[16:32].reshape(4, 4), wv=m[32:].reshape(4, 4), wfc=f)
        arch, b_int, c_int = "rtl", 0, None
        for nm in ("weights.mem", "fc_weights.mem"):
            shutil.copy(os.path.join(args.mem_dir, nm), os.path.join(args.out, nm))
        kw = dict(mul_bits=16, fc_signed=False, slice_start=4, inv_mode="rtl")
    else:
        if args.demo:
            w = dict(wq=rs.normal(0, 0.35, (4, 4)), wk=rs.normal(0, 0.35, (4, 4)), wv=rs.normal(0, 0.8, (4, 4)),
                     wfc=rs.normal(0, 0.25, 16), arch=np.array(args.arch))
            if args.arch != "rtl":
                w["b"] = np.array(-0.3)
            print("[demo] 무작위 가중치로 키트를 만듭니다. 실제 모델 값이 아닙니다.")
        else:
            if not args.weights:
                sys.exit("--weights model_fp32.npz 또는 --demo 가 필요합니다.")
            z = np.load(args.weights)
            w = {k: z[k].astype(np.float64) for k in ("wq", "wk", "wv", "wfc")}
            w["arch"] = np.array(str(z["arch"]) if "arch" in z.files else "rtl")
            for k in ("b", "c"):
                if k in z.files:
                    w[k] = z[k].astype(np.float64)
        arch = str(w["arch"])
        if arch not in ("res_square", "square", "rtl"):
            sys.exit(f"이 키트의 RTL 은 res_square 구조용입니다 (가중치 arch={arch}).")
        export_mem(args.out, w)
        wi = {k: to_q88(v) for k, v in w.items() if k in ("wq", "wk", "wv", "wfc", "b", "c")}
        b_int, c_int = (int(wi["b"]) if "b" in wi else 0), wi.get("c")
        kw = dict(mul_bits=32, fc_signed=True, slice_start=3, inv_mode="full")
        print(f"구조 {arch}: weights.mem, fc_weights.mem" + (", fc_extra.mem" if arch != "rtl" else "") + " 생성")
    for nm in ("softmax_exp_rom.mem", "softmax_inv_rom.mem"):
        shutil.copy(os.path.join(args.mem_dir, nm), os.path.join(args.out, nm))
    roms = make_roms("repo", args.mem_dir)

    # ---------------- 입력 윈도우 ----------------
    parts = []
    if args.data:
        te = np.load(os.path.join(args.data, "dataset_test.npz"))
        idx = rs.choice(len(te["y"]), min(args.n_data, len(te["y"])), replace=False)
        parts.append(te["X_q88"][idx].astype(np.int64))
    parts.append(edge_windows(rs, args.n_random))
    X = np.concatenate(parts)
    X = wrap16(X)

    # ---------------- 골든 모델 (RTL 단계 순서) ----------------
    stats = {}
    q, k, v = qkv(X, wi["wq"], wi["wk"], wi["wv"], stats)
    sc = scores(q, k, stats)
    aw = softmax_q88(sc, roms, kw["mul_bits"], kw["slice_start"], kw["inv_mode"])
    ao = attn_output(aw, v, stats)
    z = fc_q88(ao, wi["wfc"], kw["fc_signed"], stats, arch=arch, b_int=b_int, c_int=c_int, xq=X)
    label = (z > THRESHOLD_Q88).astype(np.int64)
    colsum = aw.sum(axis=1)
    cause = colsum.argmax(axis=1)

    n = len(X)
    write_words(os.path.join(args.out, "vec_n.mem"), [n])
    write_words(os.path.join(args.out, "vec_x.mem"), X)
    for nm, arr in (("q", q), ("k", k), ("v", v), ("score", sc), ("aw", aw), ("ao", ao)):
        write_words(os.path.join(args.out, f"exp_{nm}.mem"), arr)
    write_words(os.path.join(args.out, "exp_z.mem"), z)
    write_words(os.path.join(args.out, "exp_label.mem"), label)
    write_words(os.path.join(args.out, "exp_cause.mem"), cause)
    print(f"윈도우 {n}개, 이상 판정 {int(label.sum())}개, 중간 wrap 발생 비율 { {k: round(v, 4) for k, v in stats.items()} }")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()