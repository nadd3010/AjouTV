import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 경로 및 재현성 설정
# -----------------------------------------------------------------------------
TRAIN_DIR = Path("train")
FAULTS_DIR = Path("train_faults")
OUTPUT_DIR = Path("processed_dataset")

RANDOM_SEED = 2026
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# -----------------------------------------------------------------------------
# 데이터 및 하드웨어 입력 규격
# -----------------------------------------------------------------------------
SENSOR_COLS = [
    "ETCHBEAMVOLTAGE",
    "IONGAUGEPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "FLOWCOOLPRESSURE",
]

NUM_SENSORS = len(SENSOR_COLS)
WINDOW_SIZE = 4

# 원본 데이터에서 1스텝은 4초이다.
STEP_SECONDS = 4

# 고장 발생 직전 2,400스텝의 라벨 구성
NORMAL_STEPS = 1800
ABNORMAL_STEPS = 600
TOTAL_STEPS = NORMAL_STEPS + ABNORMAL_STEPS

# 2,400스텝 x 4초 = 9,600초
MIN_FAULT_INTERVAL_SECONDS = TOTAL_STEPS * STEP_SECONDS

# 각 센서의 Train 범위를 UART용 unsigned 8-bit 범위로 변환한다.
SCALED_MIN = 0
SCALED_MAX = 255


def discover_file_pairs():
    """fault 파일과 대응되는 원본 train 파일 쌍을 찾는다."""
    pairs = []
    pattern = str(FAULTS_DIR / "*_train_fault_data.csv")

    for fault_path_string in sorted(glob.glob(pattern)):
        fault_path = Path(fault_path_string)
        base_name = fault_path.name.replace("_train_fault_data.csv", "")
        train_path = TRAIN_DIR / f"{base_name}_DC_train.csv"

        if not train_path.exists():
            print(
                f"[경고] 매칭되는 train 파일을 찾을 수 없습니다: "
                f"{train_path}"
            )
            continue

        pairs.append(
            {
                "base_name": base_name,
                "train_path": train_path,
                "fault_path": fault_path,
            }
        )

    return pairs


def assign_file_splits(file_pairs):
    """원본 파일을 재현 가능한 방식으로 Train/Validation/Test에 배정한다."""
    num_files = len(file_pairs)

    if num_files < 3:
        raise ValueError(
            "원본 파일 단위 Train/Validation/Test 분리를 위해서는 "
            "최소 3개의 매칭된 원본 파일이 필요합니다."
        )

    num_train = max(1, int(num_files * TRAIN_RATIO))
    num_val = max(1, int(num_files * VAL_RATIO))

    if num_train + num_val >= num_files:
        num_val = 1
        num_train = num_files - 2

    num_test = num_files - num_train - num_val

    rng = np.random.default_rng(RANDOM_SEED)
    shuffled_indices = rng.permutation(num_files)

    train_indices = set(shuffled_indices[:num_train].tolist())
    val_indices = set(
        shuffled_indices[num_train : num_train + num_val].tolist()
    )

    split_rows = []
    file_to_split = {}
    file_id_map = {}

    for file_id, pair in enumerate(file_pairs):
        if file_id in train_indices:
            split_name = "train"
        elif file_id in val_indices:
            split_name = "val"
        else:
            split_name = "test"

        base_name = pair["base_name"]
        file_to_split[base_name] = split_name
        file_id_map[base_name] = file_id

        split_rows.append(
            {
                "file_id": file_id,
                "base_name": base_name,
                "split": split_name,
                "train_file": str(pair["train_path"]),
                "fault_file": str(pair["fault_path"]),
            }
        )

    split_counts = pd.DataFrame(split_rows)["split"].value_counts()
    print("\n[원본 파일 단위 분리]")
    print(f"전체 파일: {num_files}")
    print(f"Train 파일: {int(split_counts.get('train', 0))}")
    print(f"Validation 파일: {int(split_counts.get('val', 0))}")
    print(f"Test 파일: {int(split_counts.get('test', 0))}")

    return file_to_split, file_id_map, split_rows


def select_isolated_fault_times(df_faults, base_name):
    """
    바로 직전의 실제 고장 기록과 9,600초 이상 떨어진 고장만 선택한다.

    건너뛴 고장도 previous_fault_time에 반영하므로, 선택된 고장의 직전
    9,600초 안에 다른 고장 기록이 들어가는 것을 방지한다.
    """
    fault_times = (
        df_faults["time"].dropna().astype(np.float64).sort_values().tolist()
    )

    valid_fault_times = []
    previous_fault_time = None

    for current_time in fault_times:
        if previous_fault_time is None:
            valid_fault_times.append(current_time)
        else:
            interval_seconds = current_time - previous_fault_time

            if interval_seconds >= MIN_FAULT_INTERVAL_SECONDS:
                valid_fault_times.append(current_time)
            else:
                print(
                    f"[건너뜀: 고장 간격 부족] {base_name}, "
                    f"fault_time={current_time}, "
                    f"직전 실제 고장과의 간격={interval_seconds}초 "
                    f"(< {MIN_FAULT_INTERVAL_SECONDS}초)"
                )

        # 선택 여부와 관계없이 항상 직전 실제 고장 시각을 갱신한다.
        previous_fault_time = current_time

    return valid_fault_times, len(fault_times)


def empty_split_buffer():
    return {
        "tensors": [],
        "labels": [],
        "file_ids": [],
        "event_ids": [],
        "fault_times": [],
        "window_start_times": [],
        "window_end_times": [],
    }


def extract_windows(file_pairs, file_to_split, file_id_map):
    """각 원본 파일에서 sensor x time 윈도우와 출처 메타데이터를 생성한다."""
    buffers = {
        "train": empty_split_buffer(),
        "val": empty_split_buffer(),
        "test": empty_split_buffer(),
    }

    event_rows = []
    next_event_id = 0

    for pair in file_pairs:
        base_name = pair["base_name"]
        split_name = file_to_split[base_name]
        file_id = file_id_map[base_name]

        print(
            f"\n[처리 중] {base_name} "
            f"(split={split_name}, file_id={file_id})"
        )

        df_train = pd.read_csv(pair["train_path"])
        df_faults = pd.read_csv(pair["fault_path"])

        required_train_cols = ["time", *SENSOR_COLS]
        missing_train_cols = [
            col for col in required_train_cols if col not in df_train.columns
        ]

        if missing_train_cols:
            print(
                f"[경고] {pair['train_path']}에 필요한 컬럼이 없습니다: "
                f"{missing_train_cols}"
            )
            continue

        if "time" not in df_faults.columns:
            print(f"[경고] {pair['fault_path']}에 time 컬럼이 없습니다.")
            continue

        # time 및 센서 값을 숫자로 변환한다.
        df_train["time"] = pd.to_numeric(df_train["time"], errors="coerce")
        df_faults["time"] = pd.to_numeric(df_faults["time"], errors="coerce")

        for sensor_name in SENSOR_COLS:
            df_train[sensor_name] = pd.to_numeric(
                df_train[sensor_name], errors="coerce"
            )

        # 이전 2,400행이 실제 과거 데이터가 되도록 시간순으로 정렬한다.
        df_train = (
            df_train.dropna(subset=["time"])
            .sort_values("time")
            .reset_index(drop=True)
        )
        df_faults = (
            df_faults.dropna(subset=["time"])
            .sort_values("time")
            .reset_index(drop=True)
        )

        if df_train.empty or df_faults.empty:
            print(f"[경고] {base_name}: 유효한 train 또는 fault 데이터가 없습니다.")
            continue

        valid_fault_times, total_fault_count = select_isolated_fault_times(
            df_faults, base_name
        )

        print(
            f"[고장 사건 선별] 전체 기록 {total_fault_count}개 중 "
            f"{len(valid_fault_times)}개를 추출 대상으로 선택"
        )

        train_times = df_train["time"].to_numpy(dtype=np.float64)
        last_extracted_end_pos = None

        for fault_time in valid_fault_times:
            # fault 시각과 가장 가까운 원본 센서 데이터 행을 찾는다.
            anchor_pos = int(np.abs(train_times - fault_time).argmin())
            matched_train_time = float(train_times[anchor_pos])
            time_difference = abs(matched_train_time - fault_time)

            start_pos = anchor_pos - TOTAL_STEPS
            end_pos = anchor_pos

            if start_pos < 0:
                print(
                    f"[건너뜀: 데이터 부족] {base_name}, "
                    f"fault_time={fault_time}, 고장 이전 {TOTAL_STEPS}스텝 부족"
                )
                continue

            # 실제 원본 행 위치를 기준으로 이전 추출 구간과 중복되는지 검사한다.
            if (
                last_extracted_end_pos is not None
                and start_pos < last_extracted_end_pos
            ):
                print(
                    f"[건너뜀: 실제 행 구간 중복] {base_name}, "
                    f"fault_time={fault_time}, "
                    f"현재 구간=[{start_pos}, {end_pos}), "
                    f"이전 구간 종료={last_extracted_end_pos}"
                )
                continue

            segment = df_train.iloc[start_pos:end_pos]

            if len(segment) != TOTAL_STEPS:
                print(
                    f"[건너뜀: 추출 길이 오류] {base_name}, "
                    f"fault_time={fault_time}, 추출 길이={len(segment)}"
                )
                continue

            segment_values = segment[SENSOR_COLS].to_numpy(dtype=np.float64)

            if not np.isfinite(segment_values).all():
                print(
                    f"[건너뜀: NaN 또는 무한대] {base_name}, "
                    f"fault_time={fault_time}"
                )
                continue

            # 공식 축 정의: sensor_by_time[sensor_idx, time_idx]
            # 동일 시점의 센서 4개 값은 하나의 열을 구성한다.
            sensor_by_time = np.stack(
                [
                    segment[sensor_name].to_numpy(dtype=np.float32)
                    for sensor_name in SENSOR_COLS
                ],
                axis=0,
            )

            assert sensor_by_time.shape == (NUM_SENSORS, TOTAL_STEPS)

            # 앞쪽 1,800스텝은 정상(0), 고장 직전 600스텝은 비정상(1)
            time_labels = np.zeros(TOTAL_STEPS, dtype=np.float32)
            time_labels[NORMAL_STEPS:] = 1.0

            segment_times = segment["time"].to_numpy(dtype=np.float64)
            event_id = next_event_id
            next_event_id += 1
            num_windows = TOTAL_STEPS - WINDOW_SIZE + 1

            for start_time_idx in range(num_windows):
                end_time_idx = start_time_idx + WINDOW_SIZE

                tensor_4x4 = sensor_by_time[
                    :, start_time_idx:end_time_idx
                ]
                label_val = time_labels[end_time_idx - 1]

                assert tensor_4x4.shape == (NUM_SENSORS, WINDOW_SIZE)

                buffer = buffers[split_name]
                buffer["tensors"].append(tensor_4x4)
                buffer["labels"].append(label_val)
                buffer["file_ids"].append(file_id)
                buffer["event_ids"].append(event_id)
                buffer["fault_times"].append(fault_time)
                buffer["window_start_times"].append(
                    segment_times[start_time_idx]
                )
                buffer["window_end_times"].append(
                    segment_times[end_time_idx - 1]
                )

            event_rows.append(
                {
                    "event_id": event_id,
                    "file_id": file_id,
                    "base_name": base_name,
                    "split": split_name,
                    "fault_time": fault_time,
                    "matched_train_time": matched_train_time,
                    "time_difference_seconds": time_difference,
                    "start_pos": start_pos,
                    "end_pos_exclusive": end_pos,
                    "segment_start_time": float(segment_times[0]),
                    "segment_end_time": float(segment_times[-1]),
                    "num_windows": num_windows,
                }
            )

            # 추출과 윈도우 생성이 성공한 경우에만 갱신한다.
            last_extracted_end_pos = end_pos

            print(
                f"[추출 완료] fault_time={fault_time} -> "
                f"matched_train_time={matched_train_time}, "
                f"event_id={event_id}, windows={num_windows}"
            )

    return buffers, event_rows


def convert_buffers_to_arrays(buffers):
    """리스트 기반 버퍼를 NumPy 배열로 변환하고 기본 무결성을 검사한다."""
    arrays = {}

    for split_name, buffer in buffers.items():
        if not buffer["tensors"]:
            raise ValueError(
                f"{split_name} 분할에서 생성된 텐서가 없습니다. "
                "file_split.csv와 고장 사건 분포를 확인하세요."
            )

        split_arrays = {
            "X_raw": np.stack(buffer["tensors"], axis=0).astype(np.float32),
            "y": np.asarray(buffer["labels"], dtype=np.float32),
            "file_ids": np.asarray(buffer["file_ids"], dtype=np.int32),
            "event_ids": np.asarray(buffer["event_ids"], dtype=np.int64),
            "fault_times": np.asarray(buffer["fault_times"], dtype=np.float64),
            "window_start_times": np.asarray(
                buffer["window_start_times"], dtype=np.float64
            ),
            "window_end_times": np.asarray(
                buffer["window_end_times"], dtype=np.float64
            ),
        }

        expected_shape = (len(split_arrays["y"]), NUM_SENSORS, WINDOW_SIZE)
        if split_arrays["X_raw"].shape != expected_shape:
            raise ValueError(
                f"{split_name} X shape 오류: "
                f"실제={split_arrays['X_raw'].shape}, 예상={expected_shape}"
            )

        if not np.isfinite(split_arrays["X_raw"]).all():
            raise ValueError(f"{split_name} X_raw에 NaN 또는 무한대가 있습니다.")

        if not np.isfinite(split_arrays["y"]).all():
            raise ValueError(f"{split_name} y에 NaN 또는 무한대가 있습니다.")

        arrays[split_name] = split_arrays

    return arrays


def fit_uint8_scaler(X_train_raw):
    """Train 데이터만 사용하여 센서별 unsigned 8-bit 스케일을 계산한다."""
    sensor_min = np.empty(NUM_SENSORS, dtype=np.float64)
    sensor_max = np.empty(NUM_SENSORS, dtype=np.float64)

    for sensor_idx in range(NUM_SENSORS):
        values = X_train_raw[:, sensor_idx, :].astype(np.float64)
        sensor_min[sensor_idx] = values.min()
        sensor_max[sensor_idx] = values.max()

    sensor_range = sensor_max - sensor_min

    if np.any(sensor_range <= 0.0):
        invalid_sensors = [
            SENSOR_COLS[idx]
            for idx in np.where(sensor_range <= 0.0)[0].tolist()
        ]
        raise ValueError(
            f"Train 데이터에서 값의 범위가 0인 센서가 있습니다: "
            f"{invalid_sensors}"
        )

    scale = (SCALED_MAX - SCALED_MIN) / sensor_range

    return {
        "sensor_min": sensor_min,
        "sensor_max": sensor_max,
        "scale": scale,
    }


def apply_uint8_scaler(X_raw, scaler, split_name):
    """
    Train에서 구한 센서별 범위를 모든 분할에 동일하게 적용한다.

    scaled = round((raw - train_min) * scale)

    Train 범위를 벗어나는 Validation/Test 값은 UART의 unsigned 8-bit
    범위에 맞게 0 또는 255로 포화시킨다.
    """
    X_float = X_raw.astype(np.float64)
    output = np.empty(X_raw.shape, dtype=np.uint8)
    clipping_rows = []

    for sensor_idx, sensor_name in enumerate(SENSOR_COLS):
        values = X_float[:, sensor_idx, :]
        lower = scaler["sensor_min"][sensor_idx]
        upper = scaler["sensor_max"][sensor_idx]
        scale = scaler["scale"][sensor_idx]

        below_count = int(np.sum(values < lower))
        above_count = int(np.sum(values > upper))
        total_count = int(values.size)

        scaled = np.rint((values - lower) * scale + SCALED_MIN)
        scaled = np.clip(scaled, SCALED_MIN, SCALED_MAX)
        output[:, sensor_idx, :] = scaled.astype(np.uint8)

        clipping_rows.append(
            {
                "split": split_name,
                "sensor": sensor_name,
                "below_train_min_count": below_count,
                "above_train_max_count": above_count,
                "total_value_count": total_count,
                "clipped_ratio": (below_count + above_count) / total_count,
            }
        )

    return output, clipping_rows


def print_split_summary(split_name, split_arrays):
    X_raw = split_arrays["X_raw"]
    y = split_arrays["y"]

    normal_count = int(np.sum(y == 0.0))
    abnormal_count = int(np.sum(y == 1.0))
    abnormal_ratio = abnormal_count / len(y)

    print(f"\n[{split_name.upper()} 데이터 요약]")
    print(f"X shape: {X_raw.shape} = (sample, sensor, time)")
    print(f"y shape: {y.shape}")
    print(f"원본 파일 수: {len(np.unique(split_arrays['file_ids']))}")
    print(f"고장 사건 수: {len(np.unique(split_arrays['event_ids']))}")
    print(f"정상 윈도우 수: {normal_count}")
    print(f"비정상 윈도우 수: {abnormal_count}")
    print(f"비정상 비율: {abnormal_ratio:.6f}")
    print(f"NaN 개수: {int(np.isnan(X_raw).sum())}")
    print(f"무한대 개수: {int(np.isinf(X_raw).sum())}")

    for sensor_idx, sensor_name in enumerate(SENSOR_COLS):
        values = X_raw[:, sensor_idx, :]
        print(
            f"  {sensor_name}: "
            f"min={float(values.min()):.6f}, "
            f"max={float(values.max()):.6f}, "
            f"mean={float(values.mean()):.6f}, "
            f"std={float(values.std()):.6f}"
        )

    first_abnormal_indices = np.flatnonzero(y == 1.0)
    print("첫 번째 텐서 샘플:")
    print(X_raw[0])
    print(f"첫 번째 텐서 라벨: {y[0]}")

    if len(first_abnormal_indices) > 0:
        abnormal_idx = int(first_abnormal_indices[0])
        print(f"첫 비정상 텐서 인덱스: {abnormal_idx}")
        print(X_raw[abnormal_idx])
        print(f"첫 비정상 텐서 라벨: {y[abnormal_idx]}")


def save_outputs(arrays, scaler, file_split_rows, event_rows, clipping_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 분할과 사건의 출처를 사람이 확인할 수 있는 표로 저장한다.
    pd.DataFrame(file_split_rows).to_csv(
        OUTPUT_DIR / "file_split.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(event_rows).to_csv(
        OUTPUT_DIR / "event_metadata.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(clipping_rows).to_csv(
        OUTPUT_DIR / "scaling_clipping_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez(
        OUTPUT_DIR / "uint8_scaler_parameters.npz",
        sensor_names=np.asarray(SENSOR_COLS),
        sensor_min=scaler["sensor_min"],
        sensor_max=scaler["sensor_max"],
        scale=scaler["scale"],
        scaled_min=np.asarray(SCALED_MIN, dtype=np.int32),
        scaled_max=np.asarray(SCALED_MAX, dtype=np.int32),
    )

    summary = {
        "random_seed": RANDOM_SEED,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "axis_definition": "X[sample, sensor, time]",
        "sensor_order": SENSOR_COLS,
        "window_size": WINDOW_SIZE,
        "normal_steps_per_event": NORMAL_STEPS,
        "abnormal_steps_per_event": ABNORMAL_STEPS,
        "minimum_fault_interval_seconds": MIN_FAULT_INTERVAL_SECONDS,
        "scaled_input_dtype": "uint8",
        "scaled_input_range": [SCALED_MIN, SCALED_MAX],
        "splits": {},
    }

    for split_name, split_arrays in arrays.items():
        np.save(
            OUTPUT_DIR / f"X_{split_name}_raw.npy",
            split_arrays["X_raw"],
        )
        np.save(
            OUTPUT_DIR / f"X_{split_name}.npy",
            split_arrays["X"],
        )
        np.save(
            OUTPUT_DIR / f"y_{split_name}.npy",
            split_arrays["y"],
        )

        np.savez(
            OUTPUT_DIR / f"metadata_{split_name}.npz",
            file_ids=split_arrays["file_ids"],
            event_ids=split_arrays["event_ids"],
            fault_times=split_arrays["fault_times"],
            window_start_times=split_arrays["window_start_times"],
            window_end_times=split_arrays["window_end_times"],
        )

        y = split_arrays["y"]
        summary["splits"][split_name] = {
            "samples": int(len(y)),
            "files": int(len(np.unique(split_arrays["file_ids"]))),
            "events": int(len(np.unique(split_arrays["event_ids"]))),
            "normal_samples": int(np.sum(y == 0.0)),
            "abnormal_samples": int(np.sum(y == 1.0)),
            "abnormal_ratio": float(np.mean(y == 1.0)),
        }

    with open(OUTPUT_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    file_pairs = discover_file_pairs()

    if not file_pairs:
        raise FileNotFoundError(
            "매칭되는 train/fault CSV 파일 쌍을 찾을 수 없습니다."
        )

    file_to_split, file_id_map, file_split_rows = assign_file_splits(
        file_pairs
    )

    buffers, event_rows = extract_windows(
        file_pairs, file_to_split, file_id_map
    )
    arrays = convert_buffers_to_arrays(buffers)

    # 5개 데이터 준비 작업 중 클래스 비율, shape, NaN/무한대 및
    # 센서 범위를 각 분할에서 확인한다.
    for split_name, split_arrays in arrays.items():
        print_split_summary(split_name, split_arrays)

    # 스케일 파라미터는 Test/Validation 정보를 사용하지 않고
    # 오직 Train 데이터에서만 계산한다.
    scaler = fit_uint8_scaler(arrays["train"]["X_raw"])

    clipping_rows = []
    for split_name, split_arrays in arrays.items():
        X_scaled, split_clipping_rows = apply_uint8_scaler(
            split_arrays["X_raw"], scaler, split_name
        )
        split_arrays["X"] = X_scaled
        clipping_rows.extend(split_clipping_rows)

    save_outputs(
        arrays,
        scaler,
        file_split_rows,
        event_rows,
        clipping_rows,
    )

    print("\n[전체 데이터 준비 완료]")
    print(f"결과 폴더: {OUTPUT_DIR.resolve()}")
    print("학습 입력: X_train.npy, y_train.npy")
    print("검증 입력: X_val.npy, y_val.npy")
    print("테스트 입력: X_test.npy, y_test.npy")
    print("축 정의: X[sample, sensor, time]")
    print("스케일 입력 형식: uint8 (0~255)")
    print(
        "PyTorch에서는 X를 float32로 형변환하되 255로 다시 나누지 "
        "않아야 FPGA의 uint8 -> Q8.8 입력 경로와 값의 의미가 일치합니다."
    )


if __name__ == "__main__":
    main()