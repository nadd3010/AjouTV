import numpy as np
import torch

# ---------------------------------------------------------
# 1. 하드웨어 Q8.8 연산 모사 도구 함수
# ---------------------------------------------------------
def q88_mul(a, b):
    """Q8.8 고정소수점 곱셈 후 Q8.8로 복원 (Verilog 행렬곱 모사)"""
    # 32비트 정수 연산 후 8-bit right shift
    result = (np.int64(a) * np.int64(b)) >> 8
    return np.int32(result)

def matmul_q88(A, B):
    """행렬 곱셈 (Q8.8 포맷)"""
    # B는 전치(Transpose)된 상태로 들어온다고 가정
    rows_A = A.shape[0]
    cols_B = B.shape[1]
    result = np.zeros((rows_A, cols_B), dtype=np.int32)
    for i in range(rows_A):
        for j in range(cols_B):
            sum_val = 0
            for k in range(A.shape[1]):
                sum_val += q88_mul(A[i, k], B[k, j])
            result[i, j] = sum_val
    return result

def to_hex_string(matrix):
    """디버깅을 위한 4자리 16진수 변환"""
    # 16비트 범위로 마스킹 (음수 2의 보수 처리)
    return [f"{(val & 0xFFFF):04X}" for val in matrix.flatten()]

# ---------------------------------------------------------
# 2. 임의의 4x4 테스트 센서 데이터 및 가중치 (Q8.8 양자화 상태 가정)
# 실제로는 추출했던 W_q, W_k, W_v.mem 값들을 로드해야 합니다.
# ---------------------------------------------------------
print("[진행] Golden Model 기반 하드웨어 연산 시뮬레이션 시작...")

# 예시: Q8.8로 양자화된 4x4 입력 데이터 (256이 1.0을 의미)
X_q88 = np.array([[256, 128, 512, -256],
                  [128, 256, 256, -128],
                  [512, 128, 256,  128],
                  [-256, 128, 128, 512]], dtype=np.int32)

# 가중치 (임의의 Q8.8 값)
W_q = np.full((4, 4), 128, dtype=np.int32) # 0.5
W_k = np.full((4, 4), 256, dtype=np.int32) # 1.0
W_v = np.full((4, 4), 64, dtype=np.int32)  # 0.25
W_fc = np.full((16, 1), 25, dtype=np.int32) # FC 가중치

# ---------------------------------------------------------
# 3. 파이프라인 단계별 연산 수행 (RTL과 100% 동일한 비트 연산)
# ---------------------------------------------------------
# 단계 1: Q, K, V 변환
Q = matmul_q88(X_q88, W_q.T)
K = matmul_q88(X_q88, W_k.T)
V = matmul_q88(X_q88, W_v.T)

# 단계 2: Score 계산 및 1-bit right shift 스케일링
score_raw = matmul_q88(Q, K)
# 하드웨어 면적 최적화를 위한 1-bit right shift 적용
score_shifted = score_raw >> 1 

# 단계 3: Shifted Softmax 및 LUT 근사 (예시 로직)
# (실제로는 방금 뽑은 .mem 배열을 불러와 매핑해야 합니다)
softmax_out = np.zeros_like(score_shifted)
for i in range(4):
    local_max = np.max(score_shifted[i])
    for j in range(4):
        diff = score_shifted[i, j] - local_max
        # 예시: 차이가 0이면 100% (256), 그 외는 0으로 간주하는 극단적 LUT 모사
        softmax_out[i, j] = 256 if diff == 0 else 0 

# 단계 4: Attention Output
attn_out = matmul_q88(softmax_out, V)

# 단계 5: FC Layer 및 Argmax (분류)
flat_attn = attn_out.flatten().reshape(1, 16)
fc_out = matmul_q88(flat_attn, W_fc)
# 단일 예측 점수가 0(Threshold 예시)보다 크면 이상(1), 아니면 정상(0)
final_label = 1 if fc_out[0,0] > 0 else 0

# ---------------------------------------------------------
# 4. 기댓값 CSV 저장
# ---------------------------------------------------------
import csv

filename = "modelsim_expected_values.csv"
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Module", "Expected_Output_Hex"])
    
    writer.writerow(["Q_Matrix", " ".join(to_hex_string(Q))])
    writer.writerow(["K_Matrix", " ".join(to_hex_string(K))])
    writer.writerow(["V_Matrix", " ".join(to_hex_string(V))])
    writer.writerow(["Score_Shifted", " ".join(to_hex_string(score_shifted))])
    writer.writerow(["Softmax_LUT_Out", " ".join(to_hex_string(softmax_out))])
    writer.writerow(["Attn_Output", " ".join(to_hex_string(attn_out))])
    writer.writerow(["FC_Score", f"{(fc_out[0,0] & 0xFFFF):04X}"])
    writer.writerow(["Final_Label", f"{final_label}"])

print(f"[완료] ModelSim 교차 검증용 {filename} 생성 완료!")