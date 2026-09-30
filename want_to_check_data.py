import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. 파일 경로 설정
train_file_path = r'C:\Users\LG\Desktop\phm_data_challenge_2018\train\01_M01_DC_train.csv'

# 2. 앞서 터미널에서 확인한 실제 대문자 컬럼명 적용
target_sensors = [
    'ETCHBEAMVOLTAGE', 
    'IONGAUGEPRESSURE', 
    'ETCHGASCHANNEL1READBACK', 
    'FLOWCOOLPRESSURE'
]

# 불러올 컬럼을 'time'과 4개 센서로만 제한하여 메모리 사용량 최소화
use_cols = ['time'] + target_sensors

# 3. 전체 데이터 로드를 위한 청크 처리
chunk_size = 100000 
extracted_chunks = []

print("전체 데이터 로드 중... (시간이 조금 소요될 수 있습니다)")
for chunk in pd.read_csv(train_file_path, usecols=use_cols, chunksize=chunk_size):
    extracted_chunks.append(chunk)

# 4. 전체 조각 병합 및 시간순 정렬
full_df = pd.concat(extracted_chunks, ignore_index=True)
full_df = full_df.sort_values(by='time')

# (선택) 데이터가 너무 많아 그래프 렌더링이 느릴 경우 아래 주석을 해제하여 10개마다 1개씩 샘플링
# full_df = full_df.iloc[::10, :]

# 5. 전체 시간(time)에 대한 4개 센서 경향성 시각화
fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
fig.suptitle('전체 공정 시간(Time)에 따른 4개 센서 데이터 경향성', fontsize=16)

for i, sensor in enumerate(target_sensors):
    sensor_data = full_df[sensor]
    
    # 선 굵기와 투명도를 조절하여 미세한 변화가 잘 보이도록 설정
    axes[i].plot(full_df['time'], sensor_data, label=sensor, alpha=0.8, color=f'C{i}', linewidth=0.5)
    
    # [핵심 1] Y축 스케일을 해당 센서의 실제 최소/최대값에 타이트하게 맞춤
    y_min, y_max = sensor_data.min(), sensor_data.max()
    margin = (y_max - y_min) * 0.05 if y_max != y_min else 0.1 # 상하 5% 여백
    axes[i].set_ylim(y_min - margin, y_max + margin)
    
    # [핵심 2] Y축 눈금(Tick) 간격을 강제로 10개 내외로 촘촘하게 나누어 소수점 변화를 확인
    axes[i].yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    
    # [핵심 3] 지수 표현식(예: 1e3)이나 오프셋 사용을 방지하여 실제 값을 그대로 출력
    axes[i].ticklabel_format(useOffset=False, style='plain', axis='y')
    
    axes[i].set_ylabel('Sensor Value')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, alpha=0.3)

plt.xlabel('Time (Step)')
plt.tight_layout()
plt.show()