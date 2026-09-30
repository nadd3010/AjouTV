import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 1. 과제 계획서(3-3)에 명시된 하드웨어 친화적 초소형 모델 구조 정의
class HWAttentionModel(nn.Module):
    def __init__(self):
        super(HWAttentionModel, self).__init__()
        self.W_q = nn.Linear(4, 4, bias=False)
        self.W_k = nn.Linear(4, 4, bias=False)
        self.W_v = nn.Linear(4, 4, bias=False)
        self.W_fc = nn.Linear(16, 1, bias=False)
        
    def forward(self, x):
        pass

# 2. 통합 .npy 데이터 로드
print("[진행] 통합 데이터셋 로드 중...")
X_data = np.load("processed_tensors.npy")
y_data = np.load("processed_labels.npy")
X_tensor = torch.tensor(X_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# 3. 모델 인스턴스 생성 및 학습된 가중치 로드
model = HWAttentionModel()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# 4. 전체 데이터셋 기반 Score 분포 (백분위수) 탐색
print("[진행] 전체 데이터셋 기반 Softmax LUT Score 분포(백분위수) 탐색 중...")
all_scores = []

with torch.no_grad():
    for batch_X, _ in dataloader:
        Q = model.W_q(batch_X)
        K = model.W_k(batch_X)
        score = torch.matmul(Q, K.transpose(-2, -1)) / 2.0
        
        # 각 배치의 score 결과를 numpy 1차원 배열로 변환하여 리스트에 추가
        all_scores.append(score.cpu().numpy().flatten())

# 리스트에 모인 모든 score를 하나의 1차원 배열로 병합
scores_array = np.concatenate(all_scores)

# 5. 극단값을 제외한 실질적 분포(Percentile) 산출
print(f"\n[결과] 전체 데이터 기준 Score 절대 최솟값: {np.min(scores_array):.4f}")
print(f"[결과] 하위 0.1% 값: {np.percentile(scores_array, 0.1):.4f}")
print(f"[결과] 하위 1.0% (유효 최솟값 권장): {np.percentile(scores_array, 1.0):.4f}")
print(f"[결과] 상위 99.0% (유효 최댓값 권장): {np.percentile(scores_array, 99.0):.4f}")
print(f"[결과] 상위 99.9% 값: {np.percentile(scores_array, 99.9):.4f}")
print(f"[결과] 전체 데이터 기준 Score 절대 최댓값: {np.max(scores_array):.4f}")

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt  # 시각화를 위한 라이브러리 추가

# ... (중간 코드 동일: 모델 정의, 데이터 로드, Score 산출 배열(scores_array) 생성까지) ...

# 5. 극단값을 제외한 실질적 분포(Percentile) 산출 (기존 코드)
print(f"\n[결과] 전체 데이터 기준 Score 절대 최솟값: {np.min(scores_array):.4f}")
print(f"[결과] 하위 1.0% (유효 최솟값 권장): {np.percentile(scores_array, 1.0):.4f}")
print(f"[결과] 상위 99.0% (유효 최댓값 권장): {np.percentile(scores_array, 99.0):.4f}")
print(f"[결과] 전체 데이터 기준 Score 절대 최댓값: {np.max(scores_array):.4f}")

# ---------------------------------------------------------
# 6. [추가된 부분] 전체 Score 분포 시각화 (Histogram)
# ---------------------------------------------------------
print("\n[진행] Score 분포도 그래프를 생성합니다. 창을 닫으면 프로그램이 종료됩니다.")

plt.figure(figsize=(10, 6))

# 극단적인 이상치(수백만 단위) 때문에 그래프가 찌그러지는 것을 막기 위해,
# 하위 0.1% ~ 상위 99.9% 사이의 데이터만 잘라서 100개의 구간(bins)으로 그림
min_plot = np.percentile(scores_array, 0.1)
max_plot = np.percentile(scores_array, 99.9)
filtered_scores = scores_array[(scores_array >= min_plot) & (scores_array <= max_plot)]

# 히스토그램 그리기
# 기존 코드: plt.hist(filtered_scores, bins=100, ...)
# 수정된 코드: 함수의 반환값을 변수로 받아옵니다.
plt.hist(filtered_scores, bins=3000, color='royalblue', edgecolor='black', alpha=0.7)

plt.xlim(-250, 1400) 

plt.title("Attention Score Distribution (Zoomed In)")
plt.xlabel("Score Value")

# 하위 1%와 상위 99% 위치에 빨간색 점선으로 표시 (권장 LUT 유효 범위)
plt.axvline(np.percentile(scores_array, 1.0), color='red', linestyle='dashed', linewidth=2, label='1st Percentile (LUT Min)')
plt.axvline(np.percentile(scores_array, 99.0), color='red', linestyle='dashed', linewidth=2, label='99th Percentile (LUT Max)')

plt.title("Attention Score Distribution")
plt.xlabel("Score Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 그래프 창 띄우기
plt.show()