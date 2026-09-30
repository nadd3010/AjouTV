import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. 경로와 재현성 설정
# =============================================================================
TRAIN_DIR = Path("train")
FAULTS_DIR = Path("train_faults")

# 기존 결과를 덮어쓰지 않도록 새 폴더를 사용한다.
OUTPUT_DIR = Path("processed_dataset_v2")

RANDOM_SEED = 2026
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# =============================================================================
# 2. 센서와 시간 규격
# =============================================================================
SENSOR_COLS = [
    "ETCHBEAMVOLTAGE",
    "IONGAUGEPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "FLOWCOOLPRESSURE",
]

NUM_SENSORS = len(SENSOR_COLS)
WINDOW_SIZE = 4

# 센서 데이터의 공식 샘플링 간격
STEP_SECONDS = 4

# fault 시각과 가장 가까운 train 시각의 최대 허용 차이
MAX_FAULT_MATCH_DIFF_SECONDS = STEP_SECONDS

# 실제 관측값 사이의 간격이 12초 이하일 때만 선형보간한다.
# 관측값이 t와 t+12에 있다면 중간의 t+4, t+8 두 스텝까지 보간한다.
# 이 값보다 긴 공백을 가로지르는 보간은 허용하지 않는다.
MAX_INTERPOLATION_GAP_SECONDS = 12

# 사건 하나가 학습에 포함되기 위한 최소 유효 윈도우 수이다.
# 긴 공백을 가로지르는 윈도우를 제외한 뒤에도 정상 300개,
# 비정상 100개 이상이 남아야 해당 사건을 사용한다.
MIN_VALID_NORMAL_WINDOWS_PER_EVENT = 300
MIN_VALID_ABNORMAL_WINDOWS_PER_EVENT = 100

NORMAL_STEPS = 1800
ABNORMAL_STEPS = 600
TOTAL_STEPS = NORMAL_STEPS + ABNORMAL_STEPS

# 고장 전 2,400스텝 x 4초 = 9,600초
HISTORY_SECONDS = TOTAL_STEPS * STEP_SECONDS
MIN_FAULT_INTERVAL_SECONDS = HISTORY_SECONDS


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


def load_and_clean_train(train_path):
    """train CSV를 읽고 시간순의 유한한 센서 데이터로 정리한다."""
    df_train = pd.read_csv(train_path)
    required_cols = ["time", *SENSOR_COLS]
    missing_cols = [
        col for col in required_cols if col not in df_train.columns
    ]

    if missing_cols:
        raise ValueError(
            f"{train_path}에 필요한 컬럼이 없습니다: {missing_cols}"
        )

    for col in required_cols:
        df_train[col] = pd.to_numeric(df_train[col], errors="coerce")

    # time 또는 센서값이 NaN/inf인 행은 보간의 기준점으로 사용하지 않는다.
    df_train = df_train[required_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    df_train = df_train.dropna(subset=required_cols)
    df_train = df_train.sort_values("time").reset_index(drop=True)

    if df_train.empty:
        raise ValueError(f"{train_path}에 유효한 센서 데이터가 없습니다.")

    # 동일 time이 여러 행이면 센서별 평균으로 하나의 관측값을 만든다.
    # np.interp 계열 계산에는 time이 엄격히 증가해야 한다.
    df_train = (
        df_train.groupby("time", as_index=False, sort=True)[SENSOR_COLS]
        .mean()
        .sort_values("time")
        .reset_index(drop=True)
    )

    return df_train


def load_and_clean_faults(fault_path):
    """fault CSV의 각 행을 하나의 고장 사건으로 읽는다."""
    df_faults = pd.read_csv(fault_path)

    if "time" not in df_faults.columns:
        raise ValueError(f"{fault_path}에 time 컬럼이 없습니다.")

    df_faults["time"] = pd.to_numeric(
        df_faults["time"], errors="coerce"
    )
    df_faults = (
        df_faults.dropna(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    if df_faults.empty:
        raise ValueError(f"{fault_path}에 유효한 고장 사건이 없습니다.")

    return df_faults


def select_isolated_fault_rows(df_faults, base_name, rejection_rows):
    """
    바로 직전의 실제 고장 사건과 9,600초 이상 떨어진 사건만 선택한다.

    건너뛴 사건도 previous_fault_time에 반영한다. 따라서 선택된 사건의
    직전 9,600초 구간에 다른 fault 행이 들어가는 것을 방지한다.
    """
    selected_rows = []
    previous_fault_time = None

    for fault_row_index, row in df_faults.iterrows():
        current_time = float(row["time"])

        if previous_fault_time is None:
            selected_rows.append((fault_row_index, row))
        else:
            interval_seconds = current_time - previous_fault_time

            if interval_seconds >= MIN_FAULT_INTERVAL_SECONDS:
                selected_rows.append((fault_row_index, row))
            else:
                rejection_rows.append(
                    {
                        "base_name": base_name,
                        "fault_time": current_time,
                        "reason": "fault_interval_too_short",
                        "detail": (
                            f"previous_fault_interval={interval_seconds}, "
                            f"required={MIN_FAULT_INTERVAL_SECONDS}"
                        ),
                    }
                )

        # 선택 여부와 무관하게 직전 실제 fault 행의 시각으로 갱신한다.
        previous_fault_time = current_time

    return selected_rows


def find_nearest_time(train_times, target_time):
    """정렬된 train_times에서 target_time과 가장 가까운 값을 찾는다."""
    right_index = int(np.searchsorted(train_times, target_time, side="left"))
    candidate_indices = []

    if right_index < len(train_times):
        candidate_indices.append(right_index)
    if right_index > 0:
        candidate_indices.append(right_index - 1)

    if not candidate_indices:
        raise ValueError("가장 가까운 train time을 찾을 수 없습니다.")

    nearest_index = min(
        candidate_indices,
        key=lambda idx: abs(float(train_times[idx]) - target_time),
    )
    nearest_time = float(train_times[nearest_index])
    difference = abs(nearest_time - target_time)

    return nearest_index, nearest_time, difference


def select_evenly_spaced_indices(indices, target_count):
    """정렬된 후보에서 전체 시간 범위를 고르게 대표하도록 선택한다."""
    indices = np.asarray(indices, dtype=np.int64)

    if target_count < 0 or target_count > len(indices):
        raise ValueError(
            f"잘못된 target_count={target_count}, candidates={len(indices)}"
        )
    if target_count == 0:
        return np.empty(0, dtype=np.int64)
    if target_count == len(indices):
        return indices.copy()

    positions = (
        np.arange(target_count, dtype=np.int64) * len(indices)
    ) // target_count
    return indices[positions]


def resample_pre_fault_grid(train_times, train_values, fault_time):
    """
    고장 직전 9,600초를 4초 간격의 2,400칸 시간 격자로 만든다.

    실제 관측 간격이 12초 이하인 짧은 누락만 선형보간한다. 긴 공백이나
    보간 경계가 없는 격자점은 NaN과 invalid로 남긴다. 사건 전체를 버리지
    않고, 이후 invalid 격자점을 포함하는 4스텝 윈도우만 제외한다.
    """
    grid_times = fault_time - np.arange(
        TOTAL_STEPS, 0, -1, dtype=np.float64
    ) * STEP_SECONDS

    search_start = grid_times[0] - MAX_INTERPOLATION_GAP_SECONDS
    start_idx = int(
        np.searchsorted(train_times, search_start, side="left")
    )

    # 고장 시각과 그 이후의 센서값은 보간에 사용하지 않는다.
    end_idx = int(np.searchsorted(train_times, fault_time, side="left"))
    observed_times = train_times[start_idx:end_idx]
    observed_values = train_values[start_idx:end_idx]

    if len(observed_times) < 2:
        return None, {
            "reason": "insufficient_observations",
            "detail": f"source_observations={len(observed_times)}",
        }

    right_indices = np.searchsorted(
        observed_times, grid_times, side="left"
    )
    output = np.full(
        (NUM_SENSORS, TOTAL_STEPS), np.nan, dtype=np.float32
    )
    valid_points = np.zeros(TOTAL_STEPS, dtype=bool)
    interpolated_points = 0
    maximum_used_bracketing_gap = 0.0

    for grid_idx, (grid_time, right_idx) in enumerate(
        zip(grid_times, right_indices)
    ):
        if (
            right_idx < len(observed_times)
            and observed_times[right_idx] == grid_time
        ):
            output[:, grid_idx] = observed_values[right_idx]
            valid_points[grid_idx] = True
            continue

        left_idx = int(right_idx) - 1
        if left_idx < 0 or right_idx >= len(observed_times):
            continue

        left_time = float(observed_times[left_idx])
        right_time = float(observed_times[right_idx])
        bracketing_gap = right_time - left_time

        if (
            bracketing_gap <= 0
            or bracketing_gap > MAX_INTERPOLATION_GAP_SECONDS
        ):
            continue

        interpolation_ratio = (
            (grid_time - left_time) / bracketing_gap
        )
        interpolated_value = observed_values[left_idx] + (
            observed_values[right_idx] - observed_values[left_idx]
        ) * interpolation_ratio

        output[:, grid_idx] = interpolated_value.astype(np.float32)
        valid_points[grid_idx] = True
        interpolated_points += 1
        maximum_used_bracketing_gap = max(
            maximum_used_bracketing_gap, bracketing_gap
        )

    if not np.allclose(np.diff(grid_times), STEP_SECONDS):
        return None, {
            "reason": "grid_interval_error",
            "detail": "grid interval is not uniformly 4 seconds",
        }

    # 4개 격자점이 모두 유효한 윈도우만 사용할 수 있다.
    valid_windows = (
        np.convolve(
            valid_points.astype(np.int8),
            np.ones(WINDOW_SIZE, dtype=np.int8),
            mode="valid",
        )
        == WINDOW_SIZE
    )

    normal_boundary = NORMAL_STEPS - WINDOW_SIZE + 1
    normal_candidates = np.flatnonzero(
        valid_windows[:normal_boundary]
    )
    abnormal_candidates = np.flatnonzero(
        valid_windows[normal_boundary:]
    ) + normal_boundary

    if (
        len(normal_candidates) < MIN_VALID_NORMAL_WINDOWS_PER_EVENT
        or len(abnormal_candidates)
        < MIN_VALID_ABNORMAL_WINDOWS_PER_EVENT
    ):
        return None, {
            "reason": "insufficient_valid_windows",
            "detail": (
                f"normal={len(normal_candidates)}, "
                f"abnormal={len(abnormal_candidates)}, "
                f"required_normal={MIN_VALID_NORMAL_WINDOWS_PER_EVENT}, "
                f"required_abnormal={MIN_VALID_ABNORMAL_WINDOWS_PER_EVENT}"
            ),
        }

    # 사건별 정상:비정상 윈도우를 정확히 3:1로 맞춘다.
    selected_abnormal_count = min(
        len(abnormal_candidates), len(normal_candidates) // 3
    )
    selected_normal_count = selected_abnormal_count * 3

    selected_normal = select_evenly_spaced_indices(
        normal_candidates, selected_normal_count
    )
    selected_abnormal = select_evenly_spaced_indices(
        abnormal_candidates, selected_abnormal_count
    )
    selected_window_indices = np.sort(
        np.concatenate([selected_normal, selected_abnormal])
    )

    # 선택된 윈도우에는 NaN이 없어야 한다.
    selected_point_mask = np.zeros(TOTAL_STEPS, dtype=bool)
    for window_idx in selected_window_indices:
        selected_point_mask[window_idx : window_idx + WINDOW_SIZE] = True

    if not np.isfinite(output[:, selected_point_mask]).all():
        return None, {
            "reason": "selected_window_contains_invalid_value",
            "detail": "selected windows contain NaN or infinity",
        }

    quality = {
        "grid_times": grid_times,
        "valid_points": valid_points,
        "selected_point_mask": selected_point_mask,
        "selected_window_indices": selected_window_indices,
        "valid_normal_window_candidates": int(len(normal_candidates)),
        "valid_abnormal_window_candidates": int(
            len(abnormal_candidates)
        ),
        "selected_normal_windows": int(selected_normal_count),
        "selected_abnormal_windows": int(selected_abnormal_count),
        "interpolated_points": int(interpolated_points),
        "interpolated_ratio": interpolated_points / TOTAL_STEPS,
        "invalid_grid_points": int((~valid_points).sum()),
        "maximum_bracketing_gap_seconds": (
            maximum_used_bracketing_gap
        ),
        "source_observations": int(len(observed_times)),
    }

    return output, quality


def extract_valid_events(file_pairs):
    """모든 파일에서 시간 정합성과 연속성 검사를 통과한 사건을 추출한다."""
    valid_events = []
    rejection_rows = []
    next_event_id = 0

    for file_id, pair in enumerate(file_pairs):
        base_name = pair["base_name"]
        print(f"\n[원본 파일 처리] {base_name} (file_id={file_id})")

        try:
            df_train = load_and_clean_train(pair["train_path"])
            df_faults = load_and_clean_faults(pair["fault_path"])
        except ValueError as exc:
            print(f"[파일 건너뜀] {exc}")
            rejection_rows.append(
                {
                    "base_name": base_name,
                    "fault_time": np.nan,
                    "reason": "file_load_error",
                    "detail": str(exc),
                }
            )
            continue

        train_times = df_train["time"].to_numpy(dtype=np.float64)
        train_values = df_train[SENSOR_COLS].to_numpy(dtype=np.float32)

        selected_fault_rows = select_isolated_fault_rows(
            df_faults, base_name, rejection_rows
        )

        accepted_in_file = 0

        for fault_row_index, fault_row in selected_fault_rows:
            fault_time = float(fault_row["time"])
            _, nearest_train_time, time_difference = find_nearest_time(
                train_times, fault_time
            )

            if time_difference > MAX_FAULT_MATCH_DIFF_SECONDS:
                rejection_rows.append(
                    {
                        "base_name": base_name,
                        "fault_time": fault_time,
                        "reason": "fault_time_match_failed",
                        "detail": (
                            f"nearest_train_time={nearest_train_time}, "
                            f"difference={time_difference}, "
                            f"allowed={MAX_FAULT_MATCH_DIFF_SECONDS}"
                        ),
                    }
                )
                continue

            sensor_by_time, quality = resample_pre_fault_grid(
                train_times, train_values, fault_time
            )

            if sensor_by_time is None:
                rejection_rows.append(
                    {
                        "base_name": base_name,
                        "fault_time": fault_time,
                        "reason": quality["reason"],
                        "detail": quality["detail"],
                    }
                )
                continue

            fault_name = (
                str(fault_row["fault_name"])
                if "fault_name" in fault_row.index
                else ""
            )
            tool = (
                str(fault_row["Tool"])
                if "Tool" in fault_row.index
                else ""
            )

            valid_events.append(
                {
                    "event_id": next_event_id,
                    "file_id": file_id,
                    "base_name": base_name,
                    "fault_row_index": int(fault_row_index),
                    "fault_time": fault_time,
                    # fault_name은 감사용 메타데이터일 뿐 학습 라벨이 아니다.
                    "fault_name": fault_name,
                    "tool": tool,
                    "nearest_train_time": nearest_train_time,
                    "time_difference_seconds": time_difference,
                    "sensor_by_time": sensor_by_time,
                    **quality,
                }
            )

            next_event_id += 1
            accepted_in_file += 1

        print(
            f"[파일 추출 결과] {base_name}: "
            f"유효 사건 {accepted_in_file}개"
        )

    return valid_events, rejection_rows


def calculate_split_sizes(num_files):
    if num_files < 3:
        raise ValueError(
            "원본 파일 단위 분리를 위해 최소 3개 파일이 필요합니다."
        )

    num_train = max(1, int(num_files * TRAIN_RATIO))
    num_val = max(1, int(num_files * VAL_RATIO))

    if num_train + num_val >= num_files:
        num_val = 1
        num_train = num_files - 2

    num_test = num_files - num_train - num_val

    return num_train, num_val, num_test


def assign_splits_by_event_count(file_pairs, valid_events):
    """
    파일 단위 독립성을 유지하면서 유효 사건 수가 70/15/15에
    가장 가까워지도록 파일을 배정한다.

    파일이 24개 이하이면 가능한 Validation/Test 조합을 모두 비교한다.
    """
    num_files = len(file_pairs)
    num_train, num_val, num_test = calculate_split_sizes(num_files)

    if num_files > 24:
        raise ValueError(
            "현재의 완전탐색 분할기는 최대 24개 파일을 지원합니다."
        )

    event_counts = np.zeros(num_files, dtype=np.int64)
    for event in valid_events:
        event_counts[event["file_id"]] += 1

    total_events = int(event_counts.sum())

    if total_events == 0:
        raise ValueError("시간 품질 검사를 통과한 고장 사건이 없습니다.")

    target_counts = {
        "train": total_events * TRAIN_RATIO,
        "val": total_events * VAL_RATIO,
        "test": total_events * TEST_RATIO,
    }

    all_indices = tuple(range(num_files))
    best_score = None
    best_assignment = None

    for val_indices in combinations(all_indices, num_val):
        val_set = set(val_indices)
        remaining_after_val = tuple(
            idx for idx in all_indices if idx not in val_set
        )

        for test_indices in combinations(remaining_after_val, num_test):
            test_set = set(test_indices)
            train_indices = tuple(
                idx
                for idx in remaining_after_val
                if idx not in test_set
            )

            train_count = int(event_counts[list(train_indices)].sum())
            val_count = int(event_counts[list(val_indices)].sum())
            test_count = int(event_counts[list(test_indices)].sum())

            # 각 분할에는 최소 1개의 유효 사건이 있어야 한다.
            if min(train_count, val_count, test_count) == 0:
                continue

            counts = {
                "train": train_count,
                "val": val_count,
                "test": test_count,
            }

            score = sum(
                (
                    (counts[name] - target_counts[name])
                    / max(target_counts[name], 1.0)
                )
                ** 2
                for name in ("train", "val", "test")
            )

            if best_score is None or score < best_score:
                best_score = score
                best_assignment = {
                    "train": set(train_indices),
                    "val": val_set,
                    "test": test_set,
                }

    if best_assignment is None:
        raise ValueError(
            "모든 분할에 유효 사건을 포함하는 파일 배정을 찾지 못했습니다."
        )

    file_to_split = {}
    split_rows = []

    for file_id, pair in enumerate(file_pairs):
        if file_id in best_assignment["train"]:
            split_name = "train"
        elif file_id in best_assignment["val"]:
            split_name = "val"
        else:
            split_name = "test"

        base_name = pair["base_name"]
        file_to_split[base_name] = split_name
        split_rows.append(
            {
                "file_id": file_id,
                "base_name": base_name,
                "split": split_name,
                "valid_event_count": int(event_counts[file_id]),
                "train_file": str(pair["train_path"]),
                "fault_file": str(pair["fault_path"]),
            }
        )

    actual_counts = {
        split_name: sum(
            row["valid_event_count"]
            for row in split_rows
            if row["split"] == split_name
        )
        for split_name in ("train", "val", "test")
    }

    print("\n[유효 사건 수 기반 원본 파일 분리]")
    for split_name in ("train", "val", "test"):
        file_count = sum(
            row["split"] == split_name for row in split_rows
        )
        event_count = actual_counts[split_name]
        event_ratio = event_count / total_events
        print(
            f"{split_name}: 파일 {file_count}개, "
            f"사건 {event_count}개, 비율 {event_ratio:.2%}"
        )

    assert sum(
        row["split"] == "train" for row in split_rows
    ) == num_train
    assert sum(
        row["split"] == "val" for row in split_rows
    ) == num_val
    assert sum(
        row["split"] == "test" for row in split_rows
    ) == num_test

    return file_to_split, split_rows


def fit_train_standardizer(valid_events, file_to_split):
    """Train 사건의 원본 센서값만 사용해 센서별 평균과 표준편차를 구한다."""
    train_segments = [
        event["sensor_by_time"][:, event["selected_point_mask"]]
        for event in valid_events
        if file_to_split[event["base_name"]] == "train"
    ]

    if not train_segments:
        raise ValueError("Train 분할에 유효한 사건이 없습니다.")

    # shape: (sensor, all_train_event_times)
    train_values = np.concatenate(train_segments, axis=1).astype(np.float64)
    sensor_mean = train_values.mean(axis=1)
    sensor_std = train_values.std(axis=1)
    sensor_min = train_values.min(axis=1)
    sensor_max = train_values.max(axis=1)

    if np.any(sensor_std <= 0.0):
        invalid_sensors = [
            SENSOR_COLS[idx]
            for idx in np.where(sensor_std <= 0.0)[0].tolist()
        ]
        raise ValueError(
            f"Train에서 표준편차가 0인 센서가 있습니다: {invalid_sensors}"
        )

    return {
        "sensor_mean": sensor_mean,
        "sensor_std": sensor_std,
        "sensor_min": sensor_min,
        "sensor_max": sensor_max,
    }


def empty_split_buffer():
    return {
        "X_raw": [],
        "X": [],
        "y": [],
        "file_ids": [],
        "event_ids": [],
        "fault_times": [],
        "window_start_times": [],
        "window_end_times": [],
    }


def build_split_arrays(valid_events, file_to_split, standardizer):
    """4x4 윈도우를 생성하고 raw와 표준화 데이터를 모두 만든다."""
    buffers = {
        "train": empty_split_buffer(),
        "val": empty_split_buffer(),
        "test": empty_split_buffer(),
    }

    mean = standardizer["sensor_mean"][None, :, None]
    std = standardizer["sensor_std"][None, :, None]

    time_labels = np.zeros(TOTAL_STEPS, dtype=np.float32)
    time_labels[NORMAL_STEPS:] = 1.0
    window_labels = time_labels[WINDOW_SIZE - 1 :]

    for event in valid_events:
        split_name = file_to_split[event["base_name"]]
        sensor_by_time = event["sensor_by_time"]
        grid_times = event["grid_times"]

        # (sensor, time-window, window-size) -> (sample, sensor, time)
        all_raw_windows = np.lib.stride_tricks.sliding_window_view(
            sensor_by_time,
            window_shape=WINDOW_SIZE,
            axis=1,
        ).transpose(1, 0, 2)
        selected_indices = event["selected_window_indices"]
        raw_windows = np.ascontiguousarray(
            all_raw_windows[selected_indices], dtype=np.float32
        )

        standardized_windows = (
            (raw_windows.astype(np.float64) - mean) / std
        ).astype(np.float32)

        selected_labels = window_labels[selected_indices]
        num_windows = raw_windows.shape[0]

        if raw_windows.shape != (
            len(selected_indices), NUM_SENSORS, WINDOW_SIZE
        ):
            raise ValueError(
                f"event_id={event['event_id']}의 window shape 오류: "
                f"{raw_windows.shape}"
            )

        buffer = buffers[split_name]
        buffer["X_raw"].append(raw_windows)
        buffer["X"].append(standardized_windows)
        buffer["y"].append(selected_labels.astype(np.float32))
        buffer["file_ids"].append(
            np.full(num_windows, event["file_id"], dtype=np.int32)
        )
        buffer["event_ids"].append(
            np.full(num_windows, event["event_id"], dtype=np.int64)
        )
        buffer["fault_times"].append(
            np.full(num_windows, event["fault_time"], dtype=np.float64)
        )
        buffer["window_start_times"].append(
            grid_times[selected_indices].astype(np.float64)
        )
        buffer["window_end_times"].append(
            grid_times[selected_indices + WINDOW_SIZE - 1].astype(
                np.float64
            )
        )

    arrays = {}

    for split_name, buffer in buffers.items():
        if not buffer["X"]:
            raise ValueError(f"{split_name}에 생성된 윈도우가 없습니다.")

        arrays[split_name] = {
            key: np.concatenate(value, axis=0)
            for key, value in buffer.items()
        }

        X_raw = arrays[split_name]["X_raw"]
        X = arrays[split_name]["X"]
        y = arrays[split_name]["y"]

        if X_raw.shape != X.shape:
            raise ValueError(f"{split_name}: raw/standardized shape 불일치")
        if X.shape[1:] != (NUM_SENSORS, WINDOW_SIZE):
            raise ValueError(f"{split_name}: X shape 오류 {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"{split_name}: X와 y 길이 불일치")
        if not np.isfinite(X_raw).all() or not np.isfinite(X).all():
            raise ValueError(f"{split_name}: NaN 또는 무한대가 있습니다.")
        if not np.isin(y, [0.0, 1.0]).all():
            raise ValueError(f"{split_name}: y에 0/1 이외 값이 있습니다.")

    return arrays


def build_range_shift_report(arrays, standardizer):
    """
    Validation/Test가 Train 원본 범위를 벗어난 정도를 보고하되,
    값을 자르거나 Clipping하지 않는다.
    """
    rows = []

    for split_name, split_arrays in arrays.items():
        X_raw = split_arrays["X_raw"]

        for sensor_idx, sensor_name in enumerate(SENSOR_COLS):
            values = X_raw[:, sensor_idx, :].astype(np.float64)
            train_min = standardizer["sensor_min"][sensor_idx]
            train_max = standardizer["sensor_max"][sensor_idx]
            below_count = int(np.sum(values < train_min))
            above_count = int(np.sum(values > train_max))
            total_count = int(values.size)

            rows.append(
                {
                    "split": split_name,
                    "sensor": sensor_name,
                    "train_min": train_min,
                    "train_max": train_max,
                    "split_min": float(values.min()),
                    "split_max": float(values.max()),
                    "split_mean": float(values.mean()),
                    "split_std": float(values.std()),
                    "below_train_min_count": below_count,
                    "above_train_max_count": above_count,
                    "total_value_count": total_count,
                    "outside_train_range_ratio": (
                        below_count + above_count
                    )
                    / total_count,
                    "clipping_applied": False,
                }
            )

    return rows


def build_event_metadata(valid_events, file_to_split):
    rows = []

    for event in valid_events:
        grid_times = event["grid_times"]
        rows.append(
            {
                "event_id": event["event_id"],
                "file_id": event["file_id"],
                "base_name": event["base_name"],
                "split": file_to_split[event["base_name"]],
                "fault_row_index": event["fault_row_index"],
                "fault_time": event["fault_time"],
                "fault_name_audit_only": event["fault_name"],
                "tool_audit_only": event["tool"],
                "nearest_train_time": event["nearest_train_time"],
                "time_difference_seconds": event[
                    "time_difference_seconds"
                ],
                "segment_start_time": float(grid_times[0]),
                "segment_end_time": float(grid_times[-1]),
                "segment_duration_seconds": float(
                    grid_times[-1] - grid_times[0]
                ),
                "step_seconds": STEP_SECONDS,
                "segment_steps": TOTAL_STEPS,
                "valid_normal_window_candidates": event[
                    "valid_normal_window_candidates"
                ],
                "valid_abnormal_window_candidates": event[
                    "valid_abnormal_window_candidates"
                ],
                "selected_normal_windows": event[
                    "selected_normal_windows"
                ],
                "selected_abnormal_windows": event[
                    "selected_abnormal_windows"
                ],
                "interpolated_points": event["interpolated_points"],
                "interpolated_ratio": event["interpolated_ratio"],
                "invalid_grid_points": event["invalid_grid_points"],
                "maximum_bracketing_gap_seconds": event[
                    "maximum_bracketing_gap_seconds"
                ],
                "source_observations": event["source_observations"],
                "num_windows": int(
                    len(event["selected_window_indices"])
                ),
            }
        )

    return rows


def print_and_validate_summary(arrays):
    split_file_sets = {
        split_name: set(split_arrays["file_ids"].tolist())
        for split_name, split_arrays in arrays.items()
    }

    if split_file_sets["train"] & split_file_sets["val"]:
        raise ValueError("Train과 Validation 파일 ID가 겹칩니다.")
    if split_file_sets["train"] & split_file_sets["test"]:
        raise ValueError("Train과 Test 파일 ID가 겹칩니다.")
    if split_file_sets["val"] & split_file_sets["test"]:
        raise ValueError("Validation과 Test 파일 ID가 겹칩니다.")

    for split_name, split_arrays in arrays.items():
        y = split_arrays["y"]
        normal_count = int(np.sum(y == 0.0))
        abnormal_count = int(np.sum(y == 1.0))

        print(f"\n[{split_name.upper()} 최종 요약]")
        print(f"X shape: {split_arrays['X'].shape}")
        print(f"X dtype: {split_arrays['X'].dtype}")
        print(f"원본 파일 수: {len(split_file_sets[split_name])}")
        print(
            f"유효 고장 사건 수: "
            f"{len(np.unique(split_arrays['event_ids']))}"
        )
        print(f"정상 윈도우: {normal_count}")
        print(f"비정상 윈도우: {abnormal_count}")
        print(f"비정상 비율: {abnormal_count / len(y):.4%}")
        print(f"NaN 개수: {int(np.isnan(split_arrays['X']).sum())}")
        print(f"무한대 개수: {int(np.isinf(split_arrays['X']).sum())}")


def save_outputs(
    arrays,
    standardizer,
    split_rows,
    event_rows,
    rejection_rows,
    range_shift_rows,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(split_rows).to_csv(
        OUTPUT_DIR / "file_split.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(event_rows).to_csv(
        OUTPUT_DIR / "event_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(rejection_rows).to_csv(
        OUTPUT_DIR / "rejection_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(range_shift_rows).to_csv(
        OUTPUT_DIR / "range_shift_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.savez(
        OUTPUT_DIR / "standardization_parameters.npz",
        sensor_names=np.asarray(SENSOR_COLS),
        sensor_mean=standardizer["sensor_mean"],
        sensor_std=standardizer["sensor_std"],
        sensor_min=standardizer["sensor_min"],
        sensor_max=standardizer["sensor_max"],
    )

    summary = {
        "random_seed": RANDOM_SEED,
        "split_ratios_target": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "axis_definition": "X[sample, sensor, time]",
        "sensor_order": SENSOR_COLS,
        "window_size": WINDOW_SIZE,
        "step_seconds": STEP_SECONDS,
        "max_fault_match_diff_seconds": (
            MAX_FAULT_MATCH_DIFF_SECONDS
        ),
        "max_interpolation_gap_seconds": (
            MAX_INTERPOLATION_GAP_SECONDS
        ),
        "minimum_valid_normal_windows_per_event": (
            MIN_VALID_NORMAL_WINDOWS_PER_EVENT
        ),
        "minimum_valid_abnormal_windows_per_event": (
            MIN_VALID_ABNORMAL_WINDOWS_PER_EVENT
        ),
        "history_seconds": HISTORY_SECONDS,
        "normal_steps_per_event": NORMAL_STEPS,
        "abnormal_steps_per_event": ABNORMAL_STEPS,
        "clipping_applied": False,
        "X_files": (
            "float32 standardized with Train-only mean/std; no clipping"
        ),
        "X_raw_files": "float32 original/resampled sensor values",
        "splits": {},
        "rejections_by_reason": {},
    }

    rejection_df = pd.DataFrame(rejection_rows)
    if not rejection_df.empty:
        summary["rejections_by_reason"] = {
            str(key): int(value)
            for key, value in rejection_df["reason"].value_counts().items()
        }

    for split_name, split_arrays in arrays.items():
        np.save(
            OUTPUT_DIR / f"X_{split_name}.npy",
            split_arrays["X"].astype(np.float32),
        )
        np.save(
            OUTPUT_DIR / f"X_{split_name}_raw.npy",
            split_arrays["X_raw"].astype(np.float32),
        )
        np.save(
            OUTPUT_DIR / f"y_{split_name}.npy",
            split_arrays["y"].astype(np.float32),
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

    with open(
        OUTPUT_DIR / "dataset_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    file_pairs = discover_file_pairs()

    if not file_pairs:
        raise FileNotFoundError(
            "매칭되는 train/fault CSV 파일 쌍을 찾을 수 없습니다."
        )

    print(f"[발견된 원본 파일 쌍] {len(file_pairs)}개")

    # 1) 분할 전에 모든 파일의 유효 사건 수를 먼저 확정한다.
    valid_events, rejection_rows = extract_valid_events(file_pairs)
    print(f"\n[전체 유효 사건] {len(valid_events)}개")

    # 2) 파일 독립성을 유지하며 사건 수 70/15/15에 가깝게 배정한다.
    file_to_split, split_rows = assign_splits_by_event_count(
        file_pairs, valid_events
    )

    # 3) Train 원본 센서값으로만 표준화 파라미터를 계산한다.
    standardizer = fit_train_standardizer(
        valid_events, file_to_split
    )

    # 4) raw와 표준화 4x4 윈도우를 생성한다.
    arrays = build_split_arrays(
        valid_events, file_to_split, standardizer
    )

    # 5) 정보 손실을 유발하는 Clipping은 하지 않고 범위 차이만 보고한다.
    range_shift_rows = build_range_shift_report(
        arrays, standardizer
    )
    event_rows = build_event_metadata(
        valid_events, file_to_split
    )

    print_and_validate_summary(arrays)

    save_outputs(
        arrays,
        standardizer,
        split_rows,
        event_rows,
        rejection_rows,
        range_shift_rows,
    )

    print("\n[전처리 V2 완료]")
    print(f"결과 폴더: {OUTPUT_DIR.resolve()}")
    print("X_*.npy: Train 평균/표준편차로 표준화된 float32")
    print("X_*_raw.npy: 4초 격자로 정리된 원본 float32")
    print("Clipping: 적용하지 않음")
    print("학습 라벨: 정상(0)/고장(1) 이진 분류")
    print("fault_name: 감사용 메타데이터에만 저장, 학습에는 미사용")


if __name__ == "__main__":
    main()