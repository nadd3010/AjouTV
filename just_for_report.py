import os
import glob
import numpy as np
import pandas as pd

TRAIN_DIR = "train"
FAULTS_DIR = "train_faults"

all_tensors = []
all_labels = []

fault_files = glob.glob(os.path.join(FAULTS_DIR, "*_train_fault_data.csv"))

for fault_file in fault_files:
    base_name = os.path.basename(fault_file).replace("_train_fault_data.csv", "")
    train_file = os.path.join(TRAIN_DIR, f"{base_name}_DC_train.csv")
    
    if not os.path.exists(train_file):
        print(f"[경고] 매칭되는 train 파일을 찾을 수 없습니다: {train_file}")
        continue
        
    print(f"[처리 중] 장비 데이터 로드 및 전처리: {base_name}")
    
    df_faults = pd.read_csv(fault_file)
    df_train = pd.read_csv(train_file)
    
    valid_fault_times = []
    last_valid_time = -9999999
    
    for _, row in df_faults.iterrows():
        current_time = int(row['time'])
        if (current_time - last_valid_time) > 9600:
            valid_fault_times.append(current_time)
            last_valid_time = current_time
            
    for fault_time in valid_fault_times:
        anchor_idx = (df_train['time'] - fault_time).abs().idxmin()
        
        start_idx = anchor_idx - 2400
        end_idx = anchor_idx
        
        if start_idx >= 0 and end_idx <= len(df_train):
            sensor_cols = ['ETCHBEAMVOLTAGE', 'IONGAUGEPRESSURE', 'ETCHGASCHANNEL1READBACK', 'FLOWCOOLPRESSURE']
            window_data = df_train.iloc[start_idx:end_idx][sensor_cols].values
            
            window_labels = np.zeros(2400, dtype=np.float32)
            window_labels[1800:] = 1.0
            
            for i in range(len(window_data) - 4 + 1):
                tensor_4x4 = window_data[i : i+4, :]
                label_val = window_labels[i + 3]
                
                all_tensors.append(tensor_4x4)
                all_labels.append(label_val)

if len(all_tensors) > 0:
    X = np.array(all_tensors, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)
    
    np.save("processed_tensors.npy", X)
    np.save("processed_labels.npy", y)
    print(f"\n[전처리 완료!] 총 {len(X)}개의 4x4 텐서가 'processed_tensors.npy'로 저장되었습니다.")
else:
    print("\n[오류] 추출된 데이터가 없습니다. 폴더 경로 및 파일 내 컬럼명을 확인하세요.")









    ```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

print("[진행] 데이터셋 불러오는 중...")
X_data = np.load("processed_tensors.npy")
y_data = np.load("processed_labels.npy")

X_tensor = torch.tensor(X_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class HWAttentionModel(nn.Module):
    def __init__(self):
        super(HWAttentionModel, self).__init__()
        self.W_q = nn.Linear(4, 4, bias=False)
        self.W_k = nn.Linear(4, 4, bias=False)
        self.W_v = nn.Linear(4, 4, bias=False)
        self.W_fc = nn.Linear(16, 1, bias=False)
        
    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        score = torch.matmul(Q, K.transpose(-2, -1)) / 2.0
        
        attn_prob = torch.softmax(score, dim=-1)
        attn_out = torch.matmul(attn_prob, V)
        
        flat_out = attn_out.view(attn_out.size(0), -1)
        out = self.W_fc(flat_out)
        
        return torch.sigmoid(out)

model = HWAttentionModel()

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

def export_mem_file(tensor_weights, filename):
    quantized = torch.round(tensor_weights * 256.0).to(torch.int32)
    
    with open(filename, 'w') as f:
        for val in quantized.flatten():
            v = val.item()
            if v < 0:
                v = (1 << 16) + v
            f.write(f"{v:04X}\n")

print("[진행] 가중치 양자화 및 .mem 파일 생성 중...")

export_mem_file(model.W_q.weight.data.transpose(0, 1), "weight_Q.mem")
export_mem_file(model.W_k.weight.data.transpose(0, 1), "weight_K.mem")
export_mem_file(model.W_v.weight.data.transpose(0, 1), "weight_V.mem")
export_mem_file(model.W_fc.weight.data.transpose(0, 1), "weight_FC.mem")

print("[최종 완료] 모든 .mem 파일이 현재 폴더에 저장되었습니다!")
```
