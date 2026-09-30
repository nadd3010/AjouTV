import os
import glob

import numpy as np
import pandas as pd


TRAIN_DIR = "train"
FAULTS_DIR = "train_faults"

SENSOR_COLS = [
    "ETCHBEAMVOLTAGE",
    "IONGAUGEPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "FLOWCOOLPRESSURE",
]

NUM_SENSORS = len(SENSOR_COLS)
WINDOW_SIZE = 4

# 1스텝은 4초
STEP_SECONDS = 4

# 고장 발생 시각 이전 2,400스텝의 라벨 구성
NORMAL_STEPS = 1800
ABNORMAL_STEPS = 600
TOTAL_STEPS = NORMAL_STEPS + ABNORMAL_STEPS

# 2,400스텝 × 4초 = 9,600초
MIN_FAULT_INTERVAL_SECONDS = TOTAL_STEPS * STEP_SECONDS


all_tensors = []
all_labels = []

fault_files = sorted(
    glob.glob(
        os.path.join(
            FAULTS_DIR,
            "*_train_fault_data.csv",
        )
    )
)

for fault_file in fault_files:
    base_name = os.path.basename(fault_file).replace(
        "_train_fault_data.csv",
        "",
    )

    train_file = os.path.join(
        TRAIN_DIR,
        f"{base_name}_DC_train.csv",
    )

    if not os.path.exists(train_file):
        print(
            f"[경고] 매칭되는 train 파일을 찾을 수 없습니다: "
            f"{train_file}"
        )
        continue

    print(
        f"\n[처리 중] 장비 데이터 로드 및 전처리: "
        f"{base_name}"
    )

    df_faults = pd.read_csv(fault_file)
    df_train = pd.read_csv(train_file)

    # ---------------------------------------------------------
    # 1. 필수 컬럼 검사
    # ---------------------------------------------------------
    required_train_cols = [
        "time",
        *SENSOR_COLS,
    ]

    missing_train_cols = [
        col
        for col in required_train_cols
        if col not in df_train.columns
    ]

    if missing_train_cols:
        print(
            f"[경고] {train_file}에 필요한 컬럼이 없습니다: "
            f"{missing_train_cols}"
        )
        continue

    if "time" not in df_faults.columns:
        print(
            f"[경고] {fault_file}에 time 컬럼이 없습니다."
        )
        continue

    # ---------------------------------------------------------
    # 2. time 값을 숫자로 변환하고 시간순으로 정렬
    # ---------------------------------------------------------
    df_train["time"] = pd.to_numeric(
        df_train["time"],
        errors="coerce",
    )
    df_faults["time"] = pd.to_numeric(
        df_faults["time"],
        errors="coerce",
    )

    # train 데이터에서 time이 없는 행 제거
    df_train = (
        df_train
        .dropna(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    # fault 데이터에서 time이 없는 행 제거
    df_faults = (
        df_faults
        .dropna(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    if df_train.empty:
        print(
            f"[경고] {train_file}에 유효한 time 데이터가 없습니다."
        )
        continue

    if df_faults.empty:
        print(
            f"[경고] {fault_file}에 유효한 고장 시각이 없습니다."
        )
        continue

    # ---------------------------------------------------------
    # 3. 직전 실제 고장과 9,600초 이상 떨어진 사건만 선택
    #
    # 중요:
    # 건너뛴 고장도 previous_fault_time에 반영한다.
    # 따라서 추출 구간 내부에 다른 고장 사건이 들어가는 것을
    # 방지할 수 있다.
    # ---------------------------------------------------------
    fault_times = (
        df_faults["time"]
        .astype(np.float64)
        .tolist()
    )

    valid_fault_times = []
    previous_fault_time = None

    for current_time in fault_times:
        if previous_fault_time is None:
            valid_fault_times.append(current_time)

        else:
            interval_seconds = (
                current_time - previous_fault_time
            )

            if (
                interval_seconds
                >= MIN_FAULT_INTERVAL_SECONDS
            ):
                valid_fault_times.append(current_time)

            else:
                print(
                    f"[건너뜀: 고장 간격 부족] "
                    f"{base_name}, "
                    f"fault_time={current_time}, "
                    f"직전 고장과의 간격="
                    f"{interval_seconds}초 "
                    f"(< {MIN_FAULT_INTERVAL_SECONDS}초)"
                )

        # 추출 여부와 관계없이 직전 실제 고장 시각을 갱신
        previous_fault_time = current_time

    print(
        f"[고장 사건 선별] 전체 {len(fault_times)}개 중 "
        f"{len(valid_fault_times)}개를 추출 대상으로 선택"
    )

    # 정렬된 원본 센서 데이터의 time 배열
    train_times = df_train["time"].to_numpy(
        dtype=np.float64
    )

    # 실제로 추출된 직전 구간의 끝 위치
    last_extracted_end_pos = None

    # ---------------------------------------------------------
    # 4. 선택된 고장 사건별로 2,400스텝 추출
    # ---------------------------------------------------------
    for fault_time in valid_fault_times:
        # fault 파일의 고장 시각과 가장 가까운
        # train 데이터 행을 찾는다.
        anchor_pos = int(
            np.abs(
                train_times - fault_time
            ).argmin()
        )

        matched_train_time = train_times[anchor_pos]
        time_difference = abs(
            matched_train_time - fault_time
        )

        print(
            f"[고장 시각 매칭] "
            f"fault_time={fault_time} → "
            f"train_time={matched_train_time}, "
            f"차이={time_difference}초"
        )

        # 고장 발생 기준 행은 포함하지 않고,
        # 그 직전 2,400개 행을 추출한다.
        start_pos = anchor_pos - TOTAL_STEPS
        end_pos = anchor_pos

        # 고장 이전 데이터가 2,400스텝보다 적은 경우
        if start_pos < 0:
            print(
                f"[건너뜀: 데이터 부족] "
                f"{base_name}, "
                f"fault_time={fault_time}, "
                f"고장 이전 {TOTAL_STEPS}스텝 부족"
            )
            continue

        # 실제 행 위치를 기준으로 이전 추출 구간과
        # 겹치는지 추가 검사한다.
        if (
            last_extracted_end_pos is not None
            and start_pos < last_extracted_end_pos
        ):
            print(
                f"[건너뜀: 구간 중복] "
                f"{base_name}, "
                f"fault_time={fault_time}, "
                f"현재 구간=[{start_pos}, {end_pos}), "
                f"이전 구간 종료 위치="
                f"{last_extracted_end_pos}"
            )
            continue

        segment = df_train.iloc[
            start_pos:end_pos
        ]

        if len(segment) != TOTAL_STEPS:
            print(
                f"[건너뜀: 추출 길이 오류] "
                f"{base_name}, "
                f"fault_time={fault_time}, "
                f"추출 길이={len(segment)}, "
                f"필요 길이={TOTAL_STEPS}"
            )
            continue

        # 센서 값에 결측치가 있는지 검사
        if segment[SENSOR_COLS].isnull().any().any():
            print(
                f"[건너뜀: 센서 결측치] "
                f"{base_name}, "
                f"fault_time={fault_time}"
            )
            continue

        # -----------------------------------------------------
        # 5. sensor × time 형태로 데이터 구성
        #
        # sensor_by_time.shape:
        # (센서 수, 전체 시간 길이)
        #
        # 행: 센서
        # 열: 시간
        #
        # 동일 시점의 센서 4개 값은 한 열을 구성한다.
        # -----------------------------------------------------
        sensor_by_time = np.stack(
            [
                segment[sensor_name].to_numpy(
                    dtype=np.float32
                )
                for sensor_name in SENSOR_COLS
            ],
            axis=0,
        )

        assert sensor_by_time.shape == (
            NUM_SENSORS,
            TOTAL_STEPS,
        )

        # -----------------------------------------------------
        # 6. 시간 스텝별 라벨 생성
        #
        # 0 ~ 1799:
        # 고장 발생보다 더 이전의 정상 구간
        #
        # 1800 ~ 2399:
        # 고장 발생 직전 600스텝의 비정상 구간
        # -----------------------------------------------------
        time_labels = np.zeros(
            TOTAL_STEPS,
            dtype=np.float32,
        )
        time_labels[NORMAL_STEPS:] = 1.0

        # -----------------------------------------------------
        # 7. 시간축으로 4스텝씩 슬라이딩
        #
        # 최종 텐서 정의:
        # tensor_4x4[sensor_idx, time_idx]
        #
        # 행: 센서 4개
        # 열: 연속된 4개 시점
        # -----------------------------------------------------
        for start_time in range(
            TOTAL_STEPS - WINDOW_SIZE + 1
        ):
            end_time = (
                start_time + WINDOW_SIZE
            )

            tensor_4x4 = sensor_by_time[
                :,
                start_time:end_time,
            ]

            # 최근 4개 시점으로 마지막 시점의
            # 정상/비정상 상태를 판정한다.
            label_val = time_labels[
                end_time - 1
            ]

            assert tensor_4x4.shape == (
                NUM_SENSORS,
                WINDOW_SIZE,
            )

            all_tensors.append(tensor_4x4)
            all_labels.append(label_val)

        # 실제 추출 및 윈도우 생성에 성공한 경우에만
        # 이전 추출 구간의 끝 위치를 갱신한다.
        last_extracted_end_pos = end_pos

        print(
            f"[추출 완료] "
            f"{base_name}, "
            f"fault_time={fault_time}, "
            f"행 구간=[{start_pos}, {end_pos}), "
            f"생성 윈도우 수="
            f"{TOTAL_STEPS - WINDOW_SIZE + 1}"
        )


# -------------------------------------------------------------
# 8. 전체 데이터 저장
# -------------------------------------------------------------
if all_tensors:
    # 최종 축 정의:
    #
    # X.shape == (sample, sensor, time)
    # y.shape == (sample,)
    X = np.stack(
        all_tensors,
        axis=0,
    ).astype(np.float32)

    y = np.asarray(
        all_labels,
        dtype=np.float32,
    )

    np.save(
        "processed_tensors.npy",
        X,
    )
    np.save(
        "processed_labels.npy",
        y,
    )

    normal_count = int(
        np.sum(y == 0.0)
    )
    abnormal_count = int(
        np.sum(y == 1.0)
    )

    print("\n[전체 전처리 완료]")
    print(
        f"전체 텐서 수: {len(X)}"
    )
    print(
        f"X shape: {X.shape} "
        f"= (sample, sensor, time)"
    )
    print(
        f"y shape: {y.shape}"
    )
    print(
        f"정상 윈도우 수: {normal_count}"
    )
    print(
        f"비정상 윈도우 수: {abnormal_count}"
    )
    print(
        f"센서 행 순서: {SENSOR_COLS}"
    )
    print(
        "저장 파일: processed_tensors.npy"
    )
    print(
        "저장 파일: processed_labels.npy"
    )

else:
    print(
        "\n[오류] 추출된 데이터가 없습니다. "
        "폴더 경로, 고장 시각, 데이터 길이 및 "
        "CSV 컬럼명을 확인하세요."
    )