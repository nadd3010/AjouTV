import pandas as pd
import matplotlib.pyplot as plt

# 1. 파일 경로 및 타겟 설정
train_file_path = r'C:\Users\LG\Desktop\phm_data_challenge_2018\train\01_M01_DC_train.csv'
target_fault_time = 6533184

# 지난번 터미널 에러 해결을 반영하여 컬럼명은 대문자로 적용합니다.
target_sensors = [
    'ETCHBEAMVOLTAGE',         # 전압
    'IONGAUGEPRESSURE',        # 챔버 압력
    'ETCHGASCHANNEL1READBACK', # 공정 가스 유량
    'FLOWCOOLPRESSURE'         # 냉각 가스 압력
]

use_cols = ['time'] + target_sensors

# 2. 대용량 파일에서 목표 구간(총 2400 step)만 청크 단위로 추출
chunk_size = 100000 
extracted_chunks = []

# 분석 구간 정의
# 정상 구간: time-2400 ~ time-600 (1800 step)
# 이상 징후 구간: time-600 ~ time (600 step)
start_time = target_fault_time - 2400

print(f"고장 발생 시각({target_fault_time}) 기준 이전 2400 step 데이터 추출 중...")

for chunk in pd.read_csv(train_file_path, usecols=use_cols, chunksize=chunk_size):
    # 지정한 2400 스텝 구간 필터링
    mask = (chunk['time'] > start_time) & (chunk['time'] <= target_fault_time)
    filtered_chunk = chunk[mask]
    
    if not filtered_chunk.empty:
        extracted_chunks.append(filtered_chunk)
        
    # 고장 발생 시각을 넘어가면 불필요한 전체 탐색 조기 종료
    if chunk['time'].max() > target_fault_time:
        break

# 병합 및 시간순 정렬
full_df = pd.concat(extracted_chunks, ignore_index=True)
full_df = full_df.sort_values(by='time')

# 3. 데이터 경향성 시각화 (Subplot 및 배경색 구분)
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f'고장 발생 시각({target_fault_time}) 이전 2400 step 센서 경향성 분석', fontsize=16)

warning_start = target_fault_time - 600

for i, sensor in enumerate(target_sensors):
    axes[i].plot(full_df['time'], full_df[sensor], label=sensor, alpha=0.9, color=f'C{i}')
    
    # 정상 구간(1800 step)과 이상 징후 구간(600 step)을 배경색으로 구분
    axes[i].axvspan(start_time, warning_start, color='green', alpha=0.1, label='Normal (1800 steps)')
    axes[i].axvspan(warning_start, target_fault_time, color='orange', alpha=0.15, label='Warning (600 steps)')
    
    # 고장 시점 표시
    axes[i].axvline(x=target_fault_time, color='red', linestyle='-', linewidth=2, label='Fault Occurred')
    
    axes[i].set_ylabel('Sensor Value')
    axes[i].legend(loc='upper left')
    axes[i].grid(True, alpha=0.3)

plt.xlabel('Time (Step)')
plt.tight_layout()
plt.show()