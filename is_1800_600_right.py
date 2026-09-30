import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 파일 경로 설정
# 윈도우 경로의 백슬래시(\)로 인한 에러를 방지하기 위해 경로 문자열 앞에 r 을 붙입니다.
file_path = r"C:\Users\LG\Desktop\phm_data_challenge_2018\train\01_M01_DC_train.csv"

print("데이터를 불러오는 중입니다. (원본 데이터 크기가 커서 몇 초 정도 걸릴 수 있습니다...)")
df = pd.read_csv(file_path)

# 2. 사용자가 확인한 고장 발생 시점 설정
fault_time = 6533184

# 3. 고장 발생 시점의 데이터프레임 행 번호(Index) 자동 탐색
# 데이터프레임에서 'time' 열의 값이 6533184 인 행의 인덱스를 추출합니다.
fault_index_list = df[df['time'] == fault_time].index

if len(fault_index_list) == 0:
    print(f"오류: 파일 내에 time={fault_time} 인 데이터가 없습니다. 파일명이나 시간을 다시 확인해주세요.")
else:
    # 조건을 만족하는 첫 번째 인덱스 사용
    fault_index = fault_index_list[0]
    print(f"고장 발생 시점 탐색 완료! (행 인덱스: {fault_index})")

    # 4. 고장 시점 기준 과거 1800 step부터 고장 시점까지 데이터 슬라이싱
    # -1800 부터 0 (고장시점) 까지 총 1801개 행을 추출
    window_data = df.iloc[fault_index - 1800 : fault_index + 1]

    # 두 센서 값의 크기(Scale)를 0~1 사이로 맞추는 정규화 함수
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())

    # 5. 과제 계획서에 명시된 두 핵심 센서 데이터 추출 및 정규화
    flow_rate = normalize(window_data['FLOWCOOLFLOWRATE']).values
    pressure = normalize(window_data['FLOWCOOLPRESSURE']).values

    # 가로축(시간) 배열 생성 (-1800 ~ 0)
    steps = np.arange(-1800, 1)

    # --- 시각화(Plotting) 로직 ---
    plt.figure(figsize=(10, 5))

    # 센서 데이터 플롯
    plt.plot(steps, flow_rate, label="Cooling Flow Rate (Normalized)", color="blue", linewidth=1.5)
    plt.plot(steps, pressure, label="Chamber Pressure (Normalized)", color="orange", linewidth=1.5)

    # 정상/이상 구간 및 고장 시점 표시
    plt.axvspan(-1800, -600, color="green", alpha=0.1, label="Normal Window (1800 steps, Label 0)")
    plt.axvspan(-600, 0, color="red", alpha=0.15, label="Fault Precursor Window (600 steps, Label 1)")
    plt.axvline(0, color="red", linestyle="--", linewidth=2.5, label="Failure Event (t=0)")

    # 주석 추가 (상관관계가 틀어지기 시작하는 지점)
    plt.annotate("Correlation Drift Begins", xy=(-600, 0.5), xytext=(-1000, 0.7),
                 arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
                 fontsize=11, fontweight="bold")

    # 그래프 제목 및 라벨 설정 (사용자가 찾은 고장 유형 명시)
    plt.title(f"Precursor Analysis for 'FlowCool Pressure Dropped' (Time: {fault_time})", fontsize=14, fontweight="bold")
    plt.xlabel("Relative Time Step to Failure", fontsize=12)
    plt.ylabel("Normalized Sensor Value", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()