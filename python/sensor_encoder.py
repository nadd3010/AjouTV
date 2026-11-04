#!/usr/bin/env python
"""
PC 쪽 센서 인코더: 원본 센서 값 -> 표준화 -> Q4.4 8비트 -> UART 바이트 (시연/통합 시험용)

규약 (FPGA 쪽 input_scaler.sv, sliding_window.sv 와 맞춤)
  표준화  : x_std = clip((x_raw - mean[센서]) / std[센서], -CLIP_Z, +CLIP_Z)   mean/std 는 dataset_t/dataset_meta.json (학습과 같은 값)
            CLIP_Z = 4.0 (exp_input_range.py 결과: BEAMV 는 5σ 이상에서 알람이 꺼지고, 4σ 이하의 고정 입력은 하드웨어가 안정적)
  기울기 제한(slew): 센서 값이 시점 하나(4바이트)마다 SLEW_STEP = 0.5σ 이상 변하지 않게 한다.
            (모델은 시점 간 급변에 매우 민감하다. 4시점 안에서 2σ 이상 변하면 Q8.8 이 넘칠 수 있다 - 램프 허용치 2σ 의 25% 여유)
  인코딩  : q = clip(round(x_std * 16), -128, 127)  (부호 있는 8비트, Q4.4, 반올림. 절삭 금지)
  전송 순서: 한 시점마다 센서 4개를 이 순서로 4바이트: ETCHBEAMVOLTAGE, IONGAUGEPRESSURE, ETCHGASCHANNEL1READBACK, FLOWCOOLPRESSURE
            (sliding_window 는 센서 4개가 모이면 한 시점으로 보고 window[센서][시간] 의 마지막 열에 넣는다.
             4시점(16바이트)이 쌓이면 window_ready 가 나온다. 그 뒤에는 시점 하나(4바이트)마다 새 window.)
  FPGA    : Q8.8 원시값 = 부호 있는 바이트 x 16  (input_scaler.sv)

사용법
  python sensor_encoder.py --selftest
  python sensor_encoder.py --meta dataset_t/dataset_meta.json --demo      (표준화 값 1.0 이 어떤 바이트가 되는지 등 예시)

시리얼 전송은 pyserial 이 필요하다(선택). send_stream() 참고. 이 파일은 변환 규칙만 책임진다.
"""
import argparse
import json

import numpy as np

SENSORS = ["ETCHBEAMVOLTAGE", "IONGAUGEPRESSURE", "ETCHGASCHANNEL1READBACK", "FLOWCOOLPRESSURE"]
SCALE = 16
CLIP_Z = 4.0
SLEW_STEP = 0.5


def standardize(raw, mean, std, clip=CLIP_Z):
    """raw: (4,) 또는 (4,T) 센서 x 시점. mean/std: 길이 4."""
    raw = np.asarray(raw, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64).reshape(4, *([1] * (raw.ndim - 1)))
    s = np.asarray(std, dtype=np.float64).reshape(4, *([1] * (raw.ndim - 1)))
    return np.clip((raw - m) / s, -clip, clip)


def slew_limit(z, max_step=SLEW_STEP):
    """z: (4,T) 표준화된 값 -> 센서마다 시점 사이 변화량을 max_step 이하로 제한 (첫 시점은 그대로)."""
    z = np.asarray(z, dtype=np.float64)
    out = z.copy()
    for t in range(1, z.shape[1]):
        out[:, t] = out[:, t - 1] + np.clip(z[:, t] - out[:, t - 1], -max_step, max_step)
    return out


def prepare_stream(raw, mean, std, clip=CLIP_Z, slew=SLEW_STEP):
    """원본 센서 스트림 (4,T) -> 표준화 + 클리핑 + (선택) 기울기 제한 -> 전송할 표준화 값 (4,T). slew=None 이면 제한 없음."""
    z = standardize(raw, mean, std, clip)
    return z if slew is None else slew_limit(z, slew)


def to_q44(z):
    """표준화된 값 -> 부호 있는 Q4.4 정수(int8). 반올림 + 포화."""
    return np.clip(np.rint(np.asarray(z, dtype=np.float64) * SCALE), -128, 127).astype(np.int8)


def stream_to_bytes(z):
    """z: (4, T) 표준화된 값 -> UART 바이트열(시점마다 센서 4개). 길이 4*T."""
    q = to_q44(z)
    return bytes(int(q[s, t]) & 0xFF for t in range(q.shape[1]) for s in range(4))


def byte_to_q88(b):
    """FPGA input_scaler 와 같은 변환: 부호 있는 바이트 x 16 (16비트 two's complement 의 정수 값으로 반환)."""
    s = b - 256 if b >= 128 else b
    return s * 16


def send_stream(port, baud, z, delay_s=0.0):
    """(선택) pyserial 로 전송. 예: send_stream('COM5', 115200, z)"""
    import time

    import serial  # pip install pyserial
    data = stream_to_bytes(z)
    with serial.Serial(port, baud, timeout=1) as ser:
        for i in range(0, len(data), 4):
            ser.write(data[i:i + 4])
            if delay_s:
                time.sleep(delay_s)


class SlidingWindowModel:
    """sliding_window.sv 의 동작을 흉내 낸 모델 (바이트 순서 규약 점검용)."""

    def __init__(self):
        self.buf = [0] * 4
        self.cnt = 0
        self.filled = 0
        self.window = [[0] * 4 for _ in range(4)]            # window[센서][시간]
        self.ready = False

    def push(self, q88):
        self.ready = False
        self.buf[self.cnt] = q88
        if self.cnt == 3:
            for i in range(4):
                self.window[i] = self.window[i][1:] + [self.buf[i] if i < 3 else q88]
            self.cnt = 0
            self.filled = min(self.filled + 1, 4)
            if self.filled >= 4:
                self.ready = True
        else:
            self.cnt += 1


def selftest():
    rs = np.random.default_rng(0)
    # 1) 인코딩 라운드트립: 바이트 -> Q8.8 이 (반올림된 값) x 256 과 같아야 한다
    z = rs.uniform(-8, 8, size=(4, 50))
    data = stream_to_bytes(z)
    q88 = np.array([byte_to_q88(b) for b in data]).reshape(50, 4).T
    expect = np.rint(z * SCALE).clip(-128, 127) * (256 // SCALE)
    assert np.array_equal(q88, expect.astype(np.int64)), "라운드트립 불일치"
    # 2) 경계: 포화와 반올림
    assert to_q44(8.0) == 127 and to_q44(-8.0) == -128 and to_q44(-9.0) == -128
    assert to_q44(0.03) == 0 and to_q44(0.04) == 1 and to_q44(-0.04) == -1          # 1/32 = 0.03125 경계
    assert byte_to_q88(0x7F) == 0x07F0 and (byte_to_q88(0x80) & 0xFFFF) == 0xF800 and (byte_to_q88(0xFF) & 0xFFFF) == 0xFFF0
    # 3) 바이트 순서: 슬라이딩 윈도우 모델에 넣으면 window[센서][시간] 이 원래 (센서 x 시점) 배열과 같아야 한다
    zw = rs.uniform(-3, 3, size=(4, 4))
    sw = SlidingWindowModel()
    for b in stream_to_bytes(zw):
        sw.push(byte_to_q88(b))
    assert sw.ready, "16바이트 뒤에 window_ready 가 나와야 함"
    assert np.array_equal(np.array(sw.window), (np.rint(zw * SCALE).clip(-128, 127) * 16).astype(np.int64)), "센서/시간 순서 불일치"
    # 4) 시점 하나(4바이트)를 더 넣으면 한 칸 밀리고 새 window
    z5 = rs.uniform(-3, 3, size=(4, 5))
    sw = SlidingWindowModel()
    flags = []
    for b in stream_to_bytes(z5):
        sw.push(byte_to_q88(b))
        flags.append(sw.ready)
    assert flags.count(True) == 2 and np.array_equal(np.array(sw.window)[:, :], (np.rint(z5[:, 1:] * SCALE) * 16).astype(np.int64))
    # 5) 기울기 제한: 시점 사이 변화 <= max_step, 첫 시점 유지, 4σ 점프는 8시점에 걸쳐 도달
    zz = np.zeros((4, 12))
    zz[:, 4:] = 4.0
    lim = slew_limit(zz, 0.5)
    assert np.allclose(lim[:, 0], 0.0) and np.max(np.abs(np.diff(lim, axis=1))) <= 0.5 + 1e-12
    assert np.isclose(lim[0, 4], 0.5) and np.isclose(lim[0, 10], 3.5) and np.isclose(lim[0, 11], 4.0)     # 점프 후 8번째 시점에 4σ 도달
    # 6) prepare_stream: 클립 후 제한. 원본 값이 +10σ 여도 4σ 를 넘지 않는다
    raw = np.full((4, 20), 10.0)
    ps = prepare_stream(raw, np.zeros(4), np.ones(4), clip=4.0, slew=0.5)
    assert ps.max() <= 4.0 + 1e-12 and np.isclose(ps[0, 0], 4.0)
    print("자체 점검 통과: 라운드트립, 포화/반올림 경계, 바이트 순서(센서 x 시점), 시점 이동, 기울기 제한, 클리핑")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--meta", default=None, help="dataset_meta.json (mean/std)")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.demo:
        if not a.meta:
            raise SystemExit("--meta dataset_t/dataset_meta.json 이 필요합니다.")
        meta = json.load(open(a.meta, encoding="utf-8"))
        mean, std = np.array(meta["mean"]), np.array(meta["std"])
        print("센서 순서:", meta.get("sensors", SENSORS))
        print("표준화 상수: mean", np.round(mean, 4).tolist(), "std", np.round(std, 4).tolist())
        for k in (-1.0, 0.0, 0.5, 1.0, 2.0):
            raw = mean + k * std
            z = standardize(raw, mean, std)
            b = stream_to_bytes(z.reshape(4, 1))
            print(f"  각 센서를 평균 + {k:+.1f} 표준편차로 보내면 바이트 = {[hex(x) for x in b]}  (Q8.8 = {[hex(byte_to_q88(x) & 0xFFFF) for x in b[:1]]} ...)")
        return
    ap.print_help()


if __name__ == "__main__":
    main()