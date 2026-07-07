# 설계 결정사항 (Design Decisions)

## 공통 데이터 포맷
- 전체 시스템 고정소수점: **signed Q8.8 (16비트)**
- 음수 표현: 2의 보수(2's complement)

## Softmax LUT
- ROM 크기: **256-entry**
- 입력: score 값의 일부 비트 (인덱스 슬라이싱 위치 **미정**)
- 출력: signed Q8.8 고정소수점
- [ ] TODO: 인덱스 슬라이싱 위치 결정 필요
  - 전체 16비트 중 어느 8비트를 ROM 주소로 쓸지
  - 후보 1: score[15:8] (정수부 + 소수 상위)
  - 후보 2: score[11:4] (중간 영역)
  - Python 시뮬레이션으로 오차율 비교 후 결정

## UART TX col_sum 전송
- cause_identifier 내부: **18비트** (`[17:0]`)로 합산
- uart_tx 전송 시: **상위 2비트 버림**, `col_sum[15:0]`만 전송
- 근거: Softmax 출력 행 합 = 1.0 (Q8.8에서 0x0100)이므로 col_sum 최대 0x0400, 16비트 범위 초과 불가

## 향후 결정 필요 사항
- [ ] Softmax ROM 인덱스 슬라이싱 위치
- [ ] Weight ROM 파일 포맷 (.mem 형식 세부 규격)
- [ ] FPGA 보드 최종 선정 (Basys3 vs DE1-SoC)
