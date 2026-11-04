#!/usr/bin/env python
"""
통합(E2E) 검증 키트 생성기: UART 입력 -> input_scaler -> sliding_window -> 코어 -> UART 응답 전체를 골든 모델과 비교하기 위한 벡터

흐름
  표준화된 윈도우(데이터셋 Test) + 경계 윈도우  ->  시점마다 센서 4개를 Q4.4 바이트로 (sensor_encoder.py 규약)
  -> 시점 하나씩 보내면 슬라이딩 윈도우가 매 시점 새 window 를 만든다 (처음 3시점 이후 시점마다 1개).
  -> 각 window 에 대해 골든 모델(Q8.8, RTL 수정 4가지 적용)이 점수, 라벨, 원인, col_sum 을 계산.
  -> 기대 UART 응답 9바이트 = [ {5'b0, cause[1:0], label}, col_sum0 상위, 하위, col_sum1 상위, 하위, ... col_sum3 하위 ]

출력 폴더(kit_e2e) 안의 파일
  weights.mem, fc_weights.mem, fc_extra.mem, softmax_exp_rom.mem, softmax_inv_rom.mem   (RTL 이 읽는 파일)
  uart_bytes.mem   PC -> FPGA 로 보낼 바이트 (한 줄에 1바이트, 시점마다 센서 4개)
  e2e_n.mem        [시점 수 T, window 수 W = T - 3]
  e2e_win.mem      window 별 기대 Q8.8 값 16워드 (인덱스 = 센서*4 + 시점)
  e2e_z.mem        window 별 기대 score_abnormal (Q8.8)
  e2e_resp.mem     window 별 기대 UART 응답 9바이트
  e2e_expected.csv 사람이 보는 요약

사용법
  python make_e2e_vectors.py --weights res_square_out/model_fp32.npz --data dataset_t_hard --mem-dir ../mem --out kit_e2e
  python make_e2e_vectors.py --demo --data dataset_t_hard --mem-dir ../mem --out kit_e2e_demo      (무작위 가중치, 배선 점검용)

임계값은 argmax.sv 의 현재 값 0.5(0x0080)를 쓴다. 바꾸면 q88_golden.py 의 THRESHOLD_Q88 도 같이 바꿔야 한다.
"""
import argparse
import os
import shutil
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here), os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import q88_golden as g      # noqa: E402
import sensor_encoder as se  # noqa: E402


def hex4(v):
    return f"{int(v) & 0xFFFF:04X}"


def build_stream(Xw, n_edge, rs):
    blocks = [Xw[i] for i in range(len(Xw))]                      # 각 (4 센서, 4 시점)
    edge = [np.zeros((4, 4)), np.full((4, 4), 8.0), np.full((4, 4), -8.0),
            5.0 * np.fromfunction(lambda s, t: (-1.0) ** (s + t), (4, 4)), rs.uniform(-6, 6, (4, 4))]
    return np.concatenate(blocks + edge[:n_edge], axis=1)          # (4, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--data", required=True, help="데이터셋 폴더 (dataset_test.npz 의 표준화된 X 를 윈도우로 사용)")
    ap.add_argument("--mem-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-data", type=int, default=15, help="데이터셋에서 가져올 윈도우 수 (각 4시점)")
    ap.add_argument("--n-edge", type=int, default=5, help="경계 윈도우 수 (0~5)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rs = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ---------------- 가중치 ----------------
    if args.demo:
        w = dict(wq=rs.normal(0, 0.35, (4, 4)), wk=rs.normal(0, 0.35, (4, 4)), wv=rs.normal(0, 0.8, (4, 4)),
                 wfc=rs.normal(0, 0.25, 16), arch=np.array("res_square"), b=np.array(-0.3))
        print("[demo] 무작위 가중치입니다. 실제 모델 값이 아닙니다.")
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
        sys.exit(f"지원하지 않는 구조: {arch}")
    g.export_mem(args.out, w)
    for nm in ("softmax_exp_rom.mem", "softmax_inv_rom.mem"):
        shutil.copy(os.path.join(args.mem_dir, nm), os.path.join(args.out, nm))
    wi = {k: g.to_q88(v) for k, v in w.items() if k in g.NUM_KEYS}
    roms = g.make_roms("repo", args.mem_dir)
    kw = dict(mul_bits=32, fc_signed=True, slice_start=3, inv_mode="full", arch=arch,
              b_int=int(wi["b"]) if "b" in wi else 0, c_int=wi.get("c"))

    # ---------------- 입력 스트림 ----------------
    te = np.load(os.path.join(args.data, "dataset_test.npz"))
    idx = rs.choice(len(te["y"]), min(args.n_data, len(te["y"])), replace=False)
    Xw = te["X"][idx].astype(np.float64)                          # (n, 4 센서, 4 시점) 표준화된 값
    zs = build_stream(Xw, max(0, min(5, args.n_edge)), rs)         # (4, T)
    T = zs.shape[1]
    W = T - 3
    data = se.stream_to_bytes(zs)                                  # 길이 4T
    q88 = np.array([se.byte_to_q88(b) for b in data], dtype=np.int64).reshape(T, 4).T          # (4, T) Q8.8 정수

    # ---------------- window 구성 + 슬라이딩 윈도우 모델로 교차 점검 ----------------
    Xq = np.stack([q88[:, k:k + 4] for k in range(W)])             # (W, 4 센서, 4 시점)
    sw, ready_wins = se.SlidingWindowModel(), []
    for b in data:
        sw.push(se.byte_to_q88(b))
        if sw.ready:
            ready_wins.append(np.array(sw.window, dtype=np.int64))
    assert len(ready_wins) == W and all(np.array_equal(a, b) for a, b in zip(ready_wins, Xq)), "window 구성이 슬라이딩 윈도우 모델과 다릅니다"

    # ---------------- 골든 모델 ----------------
    out = g.forward_q88(Xq, wi["wq"], wi["wk"], wi["wv"], wi["wfc"], roms, **kw)
    score, label, cause, colsum = out["score"], out["label"], out["cause"], out["colsum"]
    resp = np.zeros((W, 9), dtype=np.int64)
    resp[:, 0] = (cause << 1) | label
    for j in range(4):
        resp[:, 1 + 2 * j] = (colsum[:, j] >> 8) & 0xFF
        resp[:, 2 + 2 * j] = colsum[:, j] & 0xFF

    # ---------------- 파일 쓰기 ----------------
    with open(os.path.join(args.out, "uart_bytes.mem"), "w") as f:
        f.write("\n".join(f"{b:02X}" for b in data) + "\n")
    with open(os.path.join(args.out, "e2e_n.mem"), "w") as f:
        f.write(f"{hex4(T)}\n{hex4(W)}\n")
    with open(os.path.join(args.out, "e2e_win.mem"), "w") as f:
        f.write("\n".join(hex4(v) for v in Xq.reshape(-1)) + "\n")
    with open(os.path.join(args.out, "e2e_z.mem"), "w") as f:
        f.write("\n".join(hex4(v) for v in score) + "\n")
    with open(os.path.join(args.out, "e2e_resp.mem"), "w") as f:
        f.write("\n".join(f"{int(v):02X}" for v in resp.reshape(-1)) + "\n")
    with open(os.path.join(args.out, "e2e_expected.csv"), "w", encoding="utf-8") as f:
        f.write("window,end_step,from_dataset,label,cause,col_sum0,col_sum1,col_sum2,col_sum3,score_q88_hex,score_float\n")
        for k in range(W):
            from_ds = int((k + 3) % 4 == 3 and (k + 3) // 4 < len(Xw))
            f.write(f"{k},{k + 3},{from_ds},{int(label[k])},{int(cause[k])},{','.join(str(int(v)) for v in colsum[k])},"
                    f"{hex4(score[k])},{score[k] / 256.0:.4f}\n")

    est_ms = W * (4 * 86.8e-6 + 0.0008) * 1e3
    print(f"시점 {T}개 -> window {W}개 (데이터셋 윈도우 {len(Xw)}개 + 경계 {max(0, min(5, args.n_edge))}개를 이어 붙임, 중간 window 는 두 윈도우가 섞임)")
    print(f"  이상 판정 {int(label.sum())}개 / 정상 {int((1 - label).sum())}개, 원인 센서 분포 {np.bincount(cause, minlength=4).tolist()}")
    print(f"  중간 wrap 발생 비율 { {k: round(v, 4) for k, v in out['stats'].items()} }")
    print(f"  시뮬레이션 시간 어림: 약 {est_ms:.0f} ms (115200 baud 기준, 시점마다 응답을 기다림)")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
