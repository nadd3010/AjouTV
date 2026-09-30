import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------
# 1. 통합 .npy 데이터 로드 및 DataLoader 세팅
# ---------------------------------------------------------
print("[진행] 데이터셋 불러오는 중...")
X_data = np.load("processed_tensors.npy")
y_data = np.load("processed_labels.npy")
# 앞서 전처리가 완료된 센서 데이터(X)와 라벨(Y)을 각각 NumPy 배열로 불러와서

X_tensor = torch.tensor(X_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1) 
# PyTorch가 학습할 수 있는 Tensor 객체로 변환함.
# 라벨 형태 맞추기

# 무작위로 섞어서 훈련시키기 위해 DataLoader에 shuffle=True 적용
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ---------------------------------------------------------
# 2. 초경량 Self-Attention 모델 설계 (하드웨어 아키텍처 반영)
# ---------------------------------------------------------
class HWAttentionModel(nn.Module):
    def __init__(self):
        super(HWAttentionModel, self).__init__()
        # 하드웨어 매핑(ROM)을 위해 Bias는 전부 False로 설정
        self.W_q = nn.Linear(4, 4, bias=False)
        self.W_k = nn.Linear(4, 4, bias=False)
        self.W_v = nn.Linear(4, 4, bias=False)
        
        # Attention 출력을 1차원으로 펼친(Flatten) 후 이진 분류를 수행할 1x16 FC Layer
        self.W_fc = nn.Linear(16, 1, bias=False)
        
    def forward(self, x):
        # 1) Q, K, V 행렬 생성
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2) Scaled Dot-Product: d_k=4이므로 루트 4인 2로 나눔 
        # (하드웨어에서는 나눗셈기 없이 1-bit right shift로 대체될 예정)
        score = torch.matmul(Q, K.transpose(-2, -1)) / 2.0
        
        # 3) Softmax 적용 후 V와 행렬곱
        attn_prob = torch.softmax(score, dim=-1)
        attn_out = torch.matmul(attn_prob, V)
        
        # 4) 1차원 Flatten 후 FC Layer 통과
        flat_out = attn_out.view(attn_out.size(0), -1)
        out = self.W_fc(flat_out)
        
        # 5) 0~1 사이의 확률값 반환
        return torch.sigmoid(out)

model = HWAttentionModel()

# ---------------------------------------------------------
# 3. 모델 학습 (BCELoss 및 Adam 옵티마이저 사용)
# ---------------------------------------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("[진행] 모델 학습 시작...")
epochs = 20
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
        
print("[완료] 모델 학습이 완료되었습니다.")

torch.save(model.state_dict(), "model_weights.pth")
print("[알림] 학습된 파이토치 모델이 model_weights.pth로 성공적으로 저장되었습니다.")

# ---------------------------------------------------------
# 4. 가중치 양자화(Q8.8) 및 .mem 파일 추출
# ---------------------------------------------------------
def export_mem_file(tensor_weights, filename):
    # 스케일 팩터(2^8 = 256) 적용하여 16비트 고정소수점(Q8.8) 정수로 변환
    quantized = torch.round(tensor_weights * 256.0).to(torch.int32)
    
    with open(filename, 'w') as f:
        for val in quantized.flatten():
            v = val.item()
            # 음수 처리를 위한 16비트 2의 보수(2's complement) 적용
            if v < 0:
                v = (1 << 16) + v
            # 16진수 문자열로 변환하여 저장
            f.write(f"{v:04X}\n")

print("[진행] 가중치 양자화 및 .mem 파일 생성 중...")
# PyTorch 모델에서 학습된 가중치를 꺼내서 함수로 전달
export_mem_file(model.W_q.weight.data.transpose(0, 1), "weight_Q.mem")
export_mem_file(model.W_k.weight.data.transpose(0, 1), "weight_K.mem")
export_mem_file(model.W_v.weight.data.transpose(0, 1), "weight_V.mem")
export_mem_file(model.W_fc.weight.data.transpose(0, 1), "weight_FC.mem")

print("[최종 완료] 모든 .mem 파일이 현재 폴더에 저장되었습니다!")
