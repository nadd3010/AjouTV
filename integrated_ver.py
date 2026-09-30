import numpy as np
import csv

print("[진행] 1. HW 아키텍처 변경 반영: 분자 LUT & 역수 ROM 추출 시작...")

# ---------------------------------------------------------
# 1. 분자 생성기 LUT (-6.0 ~ 0.0, 해상도 극대화)
# ---------------------------------------------------------
lut_min = -6.0
lut_max = 0.0
num_entries = 256

lut_inputs = np.linspace(lut_min, lut_max, num_entries)
# 분자(e^x) 값 Q8.8 양자화 (* 256)
numerator_lut = np.round(np.exp(lut_inputs) * 256.0).astype(np.int32)

with open("softmax_lut_neg6_to_0.mem", "w") as f:
    for val in numerator_lut:
        val = max(0, min(256, val)) # Clipping 방어
        f.write(f"{val:04X}\n")

# ---------------------------------------------------------
# 2. 분모 역수 곱셈기 ROM (최대 합산 1024 기준)
# ---------------------------------------------------------
rom_size = 1025
inverse_lut = np.zeros(rom_size, dtype=np.int32)

for i in range(1, rom_size):
    # 역수 Q8.8 양자화 수식: (256.0 / i) * 256.0 = 65536.0 / i
    inverse_lut[i] = min(round(65536.0 / i), 65535) # 오버플로우 방어
inverse_lut[0] = 0

with open("inverse_rom.mem", "w") as f:
    for val in inverse_lut:
        f.write(f"{val:04X}\n")

print("[완료] .mem 파일 추출 성공!")



print("\n[진행] 2. Golden Model 기반 기댓값 CSV 추출 시작...")

# ---------------------------------------------------------
# 3. 하드웨어 Q8.8 연산 모사 도구
# ---------------------------------------------------------
def q88_mul(a, b):
    return np.int32((np.int64(a) * np.int64(b)) >> 8)

def matmul_q88(A, B):
    rows_A, cols_B = A.shape[0], B.shape[1]
    result = np.zeros((rows_A, cols_B), dtype=np.int32)
    for i in range(rows_A):
        for j in range(cols_B):
            sum_val = 0
            for k in range(A.shape[1]):
                sum_val += q88_mul(A[i, k], B[k, j])
            result[i, j] = sum_val
    return result

def to_hex_string(matrix):
    return [f"{(val & 0xFFFF):04X}" for val in matrix.flatten()]

# ---------------------------------------------------------
# 4. 파이프라인 연산 (역수 곱셈 아키텍처 모사)
# ---------------------------------------------------------
X_q88 = np.array([[256, 128, 512, -256], [128, 256, 256, -128],
                  [512, 128, 256,  128], [-256, 128, 128, 512]], dtype=np.int32)
W_q = np.full((4, 4), 128, dtype=np.int32) 
W_k = np.full((4, 4), 256, dtype=np.int32) 
W_v = np.full((4, 4), 64, dtype=np.int32)  
W_fc = np.full((16, 1), 25, dtype=np.int32)

Q = matmul_q88(X_q88, W_q.T)
K = matmul_q88(X_q88, W_k.T)
V = matmul_q88(X_q88, W_v.T)

score_raw = matmul_q88(Q, K)
score_shifted = score_raw >> 1 

# [핵심] 하드웨어 Softmax(분자 LUT + 역수 ROM) 완벽 모사
softmax_out = np.zeros_like(score_shifted)
for i in range(4):
    local_max = np.max(score_shifted[i])
    numerators = np.zeros(4, dtype=np.int32)
    
    # 1단계: 분자 구하기 (LUT 참조)
    for j in range(4):
        diff_float = (score_shifted[i, j] - local_max) / 256.0 
        if diff_float < lut_min: diff_float = lut_min # Saturation 방어
        
        ratio = (diff_float - lut_min) / (lut_max - lut_min)
        idx = max(0, min(255, int(round(ratio * 255))))
        numerators[j] = numerator_lut[idx]
        
    # 2단계: 분모 합산 및 역수 ROM 참조
    denominator_sum = np.sum(numerators)
    if denominator_sum > 1024: denominator_sum = 1024
    inverse_val = inverse_lut[denominator_sum]
    
    # 3단계: 역수 곱셈 (나눗셈 대체)
    for j in range(4):
        softmax_out[i, j] = q88_mul(numerators[j], inverse_val)

attn_out = matmul_q88(softmax_out, V)
flat_attn = attn_out.flatten().reshape(1, 16)
fc_out = matmul_q88(flat_attn, W_fc)
final_label = 1 if fc_out[0,0] > 0 else 0

# ---------------------------------------------------------
# 5. ModelSim 검증용 CSV 저장
# ---------------------------------------------------------
csv_filename = "modelsim_expected_values_final.csv"
with open(csv_filename, mode='w', newline='') as file:
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

print(f"[완료] 최종 ModelSim 검증용 {csv_filename} 생성 완료! 하드웨어 팀에 전달하세요.")