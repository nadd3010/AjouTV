import numpy as np

print("[진행] HW 최적화용 역수 곱셈 ROM (inverse_rom.mem) 추출 시작...")

# 분모(4개 확률의 합)의 Q8.8 포맷 범위: 256 (최소) ~ 1024 (최대)
# 하드웨어 10-bit 주소 체계(0~1023) 및 1024까지 커버하기 위해 사이즈 1025로 설정
rom_size = 1025
inverse_lut = np.zeros(rom_size, dtype=np.int32)

for i in range(1, rom_size):
    # Q8.8 환경에서의 역수 계산 원리:
    # 1. 분모의 실제 값 X = i / 256.0
    # 2. 역수의 실제 값 Y = 1.0 / X = 256.0 / i
    # 3. 역수를 다시 Q8.8 정수로 양자화 = Y * 256.0 = (256.0 / i) * 256.0 = 65536.0 / i
    inverse_val = round(65536.0 / i)
    
    # 16비트 최대치(0xFFFF) 초과 방지 방어 로직
    if inverse_val > 65535:
        inverse_val = 65535
        
    inverse_lut[i] = inverse_val

# 0번 인덱스는 분모가 0인 예외 상황이므로 (하드웨어에서 발생 불가) 0으로 안전하게 초기화
inverse_lut[0] = 0

# 16진수 포맷으로 .mem 파일 추출
filename = "inverse_rom.mem"
with open(filename, "w") as f:
    for val in inverse_lut:
        f.write(f"{val:04X}\n")

print(f"[완료] {filename} 생성 완료! (BRAM 자원 최적화 성공)")

# --- 결과 검증 (디버깅용 출력) ---
print("\n[검증] 하드웨어 Q8.8 역수 연산 정확도 확인:")
print(f"분모 256 (실제 1.0) 일 때 역수: {inverse_lut[256]:04X} (Hex) -> 256 (Dec, 실제 1.0)")
print(f"분모 512 (실제 2.0) 일 때 역수: {inverse_lut[512]:04X} (Hex) -> 128 (Dec, 실제 0.5)")
print(f"분모 1024 (실제 4.0) 일 때 역수: {inverse_lut[1024]:04X} (Hex) -> 64 (Dec, 실제 0.25)")