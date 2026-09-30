import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

train_file_path = r'C:\Users\LG\Desktop\phm_data_challenge_2018\train\01_M01_DC_train.csv'
fault_file_path = r'C:\Users\LG\Desktop\phm_data_challenge_2018\train_faults\01_M01_train_fault_data.csv'

# 1. Fault 파일에서 원래의 고장 시각 추출 및 정렬
fault_df = pd.read_csv(fault_file_path).sort_values('time')
original_fault_times = pd.DataFrame({'time': fault_df['time']})

# 2. Train 파일에서 'time' 컬럼만 단독으로 로드 (메모리 점유율 매우 낮음)
print("Train 파일의 time 데이터 스캔 중...")
train_times = pd.read_csv(train_file_path, usecols=['time']).sort_values('time')

# 3. pd.merge_asof를 이용해 가장 가까운 시각(nearest)으로 자동 보정
matched_times = pd.merge_asof(original_fault_times, train_times, on='time', direction='nearest')
adjusted_fault_times = matched_times['time'].tolist()

print("\n[고장 발생 시각 보정 결과]")
for orig, adj in zip(original_fault_times['time'], adjusted_fault_times):
    print(f"원본 시각: {orig} -> 보정 시각: {adj} (오차: {abs(orig - adj)} 스텝)")

# 4. 보정된 시각(adjusted_fault_times)을 바탕으로 기존의 청크 추출 로직 실행
target_sensors = ['ETCHBEAMVOLTAGE', 'IONGAUGEPRESSURE', 'ETCHGASCHANNEL1READBACK', 'FLOWCOOLPRESSURE']
use_cols = ['time'] + target_sensors

extracted_data_per_fault = {ft: [] for ft in adjusted_fault_times}

print(f"\n총 {len(adjusted_fault_times)}개의 보정된 고장 케이스에 대해 데이터 추출을 시작합니다...")

chunk_size = 100000 
for chunk in pd.read_csv(train_file_path, usecols=use_cols, chunksize=chunk_size):
    for ft in adjusted_fault_times:
        start_time = ft - 3000
        mask = (chunk['time'] > start_time) & (chunk['time'] <= ft)
        filtered_chunk = chunk[mask]
        
        if not filtered_chunk.empty:
            extracted_data_per_fault[ft].append(filtered_chunk)
            
    if chunk['time'].max() > max(adjusted_fault_times):
        break

# (이후 시각화 코드는 동일하게 진행)

# 1. 파일 경로 설정
train_file_path = r'C:\Users\LG\Desktop\phm_data_challenge_2018\train\01_M01_DC_train.csv'
fault_file_path = r'C:\Users\LG\Desktop\phm_data_challenge_2018\train_faults\01_M01_train_fault_data.csv'

# 2. 타겟 센서 컬럼 설정 (대문자)
target_sensors = [
    'ETCHBEAMVOLTAGE',         # 전압
    'IONGAUGEPRESSURE',        # 챔버 압력
    'ETCHGASCHANNEL1READBACK', # 공정 가스 유량
    'FLOWCOOLPRESSURE'         # 냉각 가스 압력
]
use_cols = ['time'] + target_sensors

# 3. Fault 파일에서 모든 고장 발생 시각 및 고장 유형(옵션) 로드
fault_df = pd.read_csv(fault_file_path)
fault_times = fault_df['time'].tolist()
# fault_df에 'fault_name' 등의 컬럼이 있다면 유형도 함께 추출할 수 있습니다.

# 각 고장 시각별로 추출된 데이터를 담을 딕셔너리 준비
extracted_data_per_fault = {ft: [] for ft in fault_times}

print(f"총 {len(fault_times)}개의 고장 케이스에 대해 이전 3000 step 데이터 추출을 시작합니다...")

# 4. 대용량 Train 파일 청크 단위 탐색 (한 번의 순회로 모든 고장 케이스 추출)
chunk_size = 100000 
for chunk in pd.read_csv(train_file_path, usecols=use_cols, chunksize=chunk_size):
    for ft in fault_times:
        start_time = ft - 3000
        # 각 고장 시각별 3000스텝 구간 마스킹
        mask = (chunk['time'] > start_time) & (chunk['time'] <= ft)
        filtered_chunk = chunk[mask]
        
        if not filtered_chunk.empty:
            extracted_data_per_fault[ft].append(filtered_chunk)
            
    # 가장 마지막 고장 시각의 구간을 지나치면 더 이상 읽을 필요가 없으므로 조기 종료
    if chunk['time'].max() > max(fault_times):
        print("모든 고장 케이스 추출 완료. 탐색을 조기 종료합니다.")
        break

# 5. 각 고장 시각별로 개별 시각화 수행
for i, ft in enumerate(fault_times):
    if not extracted_data_per_fault[ft]:
        print(f"경고: 고장 시각 {ft}에 해당하는 데이터가 train 파일에 없습니다.")
        continue
        
    # 조각난 청크들을 하나의 데이터프레임으로 병합하고 시간순 정렬
    df_fault = pd.concat(extracted_data_per_fault[ft], ignore_index=True)
    df_fault = df_fault.sort_values(by='time')
    
    # 시각화 (Y축 소수점 최적화 포함)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Case {i+1}: 고장 발생 시각({ft}) 이전 3000 step 센서 경향성', fontsize=16)
    
    for j, sensor in enumerate(target_sensors):
        sensor_data = df_fault[sensor]
        axes[j].plot(df_fault['time'], sensor_data, label=sensor, alpha=0.9, color=f'C{j}')
        
        # 고장 발생 시점을 붉은 점선으로 표시
        axes[j].axvline(x=ft, color='red', linestyle='--', linewidth=2, label='Fault Occurred')
        
        # [핵심] Y축 스케일을 타이트하게 맞추어 미세한 소수점 변동 관찰
        y_min, y_max = sensor_data.min(), sensor_data.max()
        margin = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
        axes[j].set_ylim(y_min - margin, y_max + margin)
        axes[j].yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        axes[j].ticklabel_format(useOffset=False, style='plain', axis='y')
        
        axes[j].set_ylabel('Sensor Value')
        axes[j].legend(loc='upper left')
        axes[j].grid(True, alpha=0.3)

    plt.xlabel('Time (Step)')
    plt.tight_layout()
    plt.show() # 각 고장 케이스마다 개별적으로 창을 띄웁니다.