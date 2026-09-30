import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 1. 모델 정의 (과제 계획서의 초경량 하드웨어 아키텍처 반영)
class HWAttentionModel(nn.Module):
    def __init__(self):
        super(HWAttentionModel, self).__init__()
        self.W_q = nn.Linear(4, 4, bias=False)
        self.W_k = nn.Linear(4, 4, bias=False)
        self.W_v = nn.Linear(4, 4, bias=False)
        self.W_fc = nn.Linear(16, 1, bias=False)

    def forward(self, x):
        pass

# 2. 통합 데이터 로드 및 모델 준비
print("[진행] 통합 데이터셋 및 모델 가중치 로드 중...")
X_data = np.load("processed_tensors.npy")
y_data = np.load("processed_labels.npy")
X_tensor = torch.tensor(X_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

model = HWAttentionModel()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# 3. Shifted Softmax 기반 Score 추출
print("[진행] Shifted Softmax 기반 Score 분포 탐색 중...")
all_shifted_scores = []

with torch.no_grad():
    for batch_X, _ in dataloader:
        Q = model.W_q(batch_X)
        K = model.W_k(batch_X)
        # 1-bit right shift 스케일링 모사 (/ 2.0)
        raw_scores = torch.matmul(Q, K.transpose(-2, -1)) / 2.0
        
        # [핵심] 행(Row) 단위로 지역 최댓값(Local Max) 찾기
        # dim=-1은 4개의 Key에 대한 Score를 의미함
        local_max = raw_scores.max(dim=-1, keepdim=True)[0]
        
        # 원시 Score에서 최댓값 차감 (Shift) -> 이제 모든 값이 0 이하의 음수!
        shifted_scores = raw_scores - local_max
        
        all_shifted_scores.append(shifted_scores.cpu().numpy().flatten())

shifted_array = np.concatenate(all_shifted_scores)

# 4. 하위 1% 백분위수 도출 및 최종 하한값 결정
percentile_1st = np.percentile(shifted_array, 1.0)
print(f"\n[분석 결과] Shift 처리된 Score의 하위 1.0% 값: {percentile_1st:.4f}")

# Q8.8 포맷의 수학적 한계(-8.0)를 고려하여 최종 LUT 하한값 결정
# 하위 1%가 -8.0보다 낮더라도, 하드웨어 연산 규격상 0으로 수렴하므로 -8.0으로 포화(Saturation)
lut_min = max(percentile_1st, -8.0)
lut_max = 0.0
print(f"[설정] 256칸 LUT 생성을 위한 최종 유효 구간: {lut_min:.4f} ~ {lut_max:.4f}\n")

# 5. 유효 구간 256칸(8-bit index) 분할 및 지수 연산
lut_inputs = np.linspace(lut_min, lut_max, 256)
exp_values = np.exp(lut_inputs)

# 6. 스케일링 및 양자화 (Q8.8 포맷)
# 16비트 고정소수점 정수로 양자화하기 위해 스케일 팩터(2^8 = 256) 적용
scale_factor = 256.0
quantized_lut = np.round(exp_values * scale_factor).astype(np.int32)

# 7. Clipping 및 16진수 .mem 파일 추출
filename = "softmax_lut_shifted_final.mem"
with open(filename, "w") as f:
    for val in quantized_lut:
        # 안전을 위한 추가 Clipping: 최대 확률 1.0(256) 보장
        if val < 0: val = 0
        elif val > 256: val = 256
        f.write(f"{val:04X}\n")

print(f"[최종 완료] Xilinx Basys 3 보드 이식을 위한 {filename} 파일이 성공적으로 생성되었습니다!")