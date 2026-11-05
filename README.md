# AjouTV
HW Attention 기반 반도체 공정 이상 AI 감지 센서

식각 장비 센서 4개(4시점)를 PC가 UART로 보내면, Basys3 FPGA 위의 Q8.8 고정소수점 Self-Attention 분류기가 이상 여부와 기여 센서를 판정해 LED, 버저, UART 응답으로 돌려준다.
이 문서는 **최종 모델(res_square)을 반영한 기준(2026-11-04)** 이다.

## 1. 한눈에 보기

| 항목 | 내용 |
| --- | --- |
| 입력 | 센서 4개 x 최근 4시점 = 4x4 행렬. 센서 순서: ETCHBEAMVOLTAGE, IONGAUGEPRESSURE, ETCHGASCHANNEL1READBACK, FLOWCOOLPRESSURE |
| 수치 형식 | 내부 signed Q8.8(16비트), 입력은 PC가 표준화 후 **부호 있는 Q4.4 8비트**(x16, 반올림)로 전송 |
| 모델 | 4x4 Self-Attention(Wq, Wk, Wv 각 4x4) + 머리 `z = Σ w·(O + X)² + b` (잔차 + 제곱, 파라미터 65개) |
| 판정 | `z > 0.5`(`16'h0080`)이면 이상. 원인 센서는 attention 열합의 argmax(참고용 추정) |
| UART | 115200 baud 8N1, 입력은 시점마다 센서 4바이트, 응답은 9바이트 |
| 클럭 | 현재 RTL은 50MHz 기준(`CLKS_PER_BIT = 434`). Basys3의 100MHz 대응은 별도 작업(미반영) |
| 지연 | window_ready -> classify_done 114클럭(ModelSim 측정) |

**데이터 흐름**
```
PC(sensor_encoder.py: 표준화, 클리핑, 기울기 제한, Q4.4)
 -> UART -> uart_rx -> input_scaler(Q4.4 -> Q8.8) -> sliding_window(4x4)
 -> qkv_transform -> score_compute -> softmax_lut -> attn_output
 -> fc_layer(res_square) -> argmax(label) / cause_identifier(원인)
 -> led_controller / buzzer_driver / uart_tx(9바이트 응답)
```
UART 응답 형식: `byte0 = {5'b0, cause_index[1:0], label}`, 이어서 `col_sum[0..3]`을 각각 상위, 하위 바이트(16비트)로 전송.

## 2. 디렉터리 구조

`★` = 최종 모델 반영을 위해 이번에 수정, `+` = 이번에 추가한 파일.
```
AjouTV/
├── rtl/                           # 합성 대상 RTL (SystemVerilog, .sv)
│   ├── top.sv                     ★ 최상위 (fc_layer 에 .x(window) 연결 한 줄 추가)
│   ├── sensor_interface/          #   uart_rx.sv, input_scaler.sv ★
│   ├── sliding_window/            #   sliding_window.sv
│   ├── attention_engine/          #   qkv_transform.sv, score_compute.sv, softmax_lut.sv ★, attn_output.sv
│   ├── classifier/                #   fc_layer.sv ★, argmax.sv, cause_identifier.sv
│   ├── control_fsm/               #   main_sequencer.sv
│   ├── output_interface/          #   led_controller.sv, buzzer_driver.sv, uart_tx.sv
│   └── schematic.pdf
├── mem/                           # ROM 초기값: 최종 가중치 (3절)
├── tb/                            # 테스트벤치 (4절)
├── sim/                           # ModelSim .do 스크립트 (4절)
├── python/                        # 학습, 골든 모델, 검증 벡터 생성, PC 인코더 (5절)
│   └── final_model/               + 최종 모델 산출물(가중치, .mem, 표준화 상수)
├── constraints/                   # FPGA 핀 배치(.xdc) — 아직 없음
├── docs/                          # 설계 결정, 인터페이스 정의
├── .gitignore
└── README.md
```

## 3. rtl 모듈과 mem

### rtl 모듈 구조
| 파일 | 역할 | 비고 |
| --- | --- | --- |
| `sensor_interface/uart_rx.sv` | UART 수신(115200, 8N1), 2단 FF로 동기화, 바이트마다 `data_valid` | |
| `sensor_interface/input_scaler.sv` | 부호 있는 Q4.4 바이트를 Q8.8로 복원: `{{4{b[7]}}, b, 4'b0}` | ★ 기존은 `{b, 8'b0}`(부호 없는 정수) |
| `sliding_window/sliding_window.sv` | 센서 4바이트가 모이면 한 시점으로 보고 4x4 윈도우를 이동, `window_ready` 펄스 | 바이트 순서: 시점마다 센서 0~3 |
| `attention_engine/qkv_transform.sv` | `X x Wq/Wk/Wv` -> Q, K, V (Q8.8, [23:8]) | `weights.mem` 48워드 |
| `attention_engine/score_compute.sv` | `Q x Kᵀ`, √dk=2를 `>>>1`로 처리 | |
| `attention_engine/softmax_lut.sv` | 행별 softmax 근사(exp ROM 256칸, 역수 ROM 1025칸, 곱셈) | ★ 곱 32비트, 역수 ROM 인덱스, `SLICE_START=3` 정합 수정 |
| `attention_engine/attn_output.sv` | `A x V` | |
| `classifier/fc_layer.sv` | `z = Σ w·(O + X)² + b` (FC 가중치 16개 + 바이어스) | ★ 부호 있는 가중치, 잔차 + 제곱 + 바이어스 |
| `classifier/argmax.sv` | `z > 0.5` 이면 `label = 1` (부호 있는 비교) | 임계값 `16'h0080` |
| `classifier/cause_identifier.sv` | attention 열합이 가장 큰 센서의 인덱스(`cause_index`) | 참고용 추정 |
| `control_fsm/main_sequencer.sv` | Moore FSM으로 `ATTENTION -> CLASSIFY -> OUTPUT` 단계 제어 | |
| `output_interface/led_controller.sv`, `buzzer_driver.sv` | 이상 시 원인 센서와 1:1 LED, 버저 | |
| `output_interface/uart_tx.sv` | 9바이트 응답 송신 | 송신 중 들어온 window는 무시됨 |
| `top.sv` | 위 모듈 연결 | ★ |

### mem (최종 가중치, 합성과 시뮬레이션이 읽는 ROM 초기값)
| 파일 | 내용 |
| --- | --- |
| `weights.mem` | Wq, Wk, Wv (48워드, Q8.8) |
| `fc_weights.mem` | FC 가중치 (16워드) |
| `fc_extra.mem` | FC 바이어스 (+ 새 파일: **Vivado 설계 소스에 추가 필요**) |
| `softmax_exp_rom.mem`, `softmax_inv_rom.mem` | softmax exp ROM(256), 역수 ROM(1025) |

세 가중치 파일은 `python/final_model/`의 같은 이름 파일과 내용이 같아야 한다(확인 방법은 6절).

## 4. 테스트벤치와 sim 스크립트

### 최종 검증에 쓰는 것
| 파일 | 무엇을 검증하나 |
| --- | --- |
| `tb/tb_input_scaler.sv` + `sim/run_input_scaler.do` | Q4.4 -> Q8.8 변환 9건과 `scaled_valid` 펄스 |
| `tb/tb_core_e2e.sv` + `tb/tb_fc_layer.sv` + `tb/tb_softmax_lut_vec.sv` + `sim/run_kit.do` | 코어 데이터패스를 골든 모델과 단계별 비교(q, k, v, score, attn_w, attn_out, z, label, cause) 216 윈도우, `fc_layer`와 `softmax_lut` 단위 시험 |
| `tb/tb_top_e2e.sv` + `sim/run_e2e.do` | **UART 입력부터 UART 응답까지** top 전체: 입력 경로, 코어, 출력 경로를 window마다 비교 |

### 그 밖의 파일
| 파일 | 설명 |
| --- | --- |
| `tb/tb_top.sv` + `sim/run_top.do` | top 구동 스모크 테스트(어서션 없음, 모델 입력 범위 밖의 값 사용). 최종 판정에는 쓰지 않음 |
| `tb/tb_uart_rx.sv`, `tb_uart_tx.sv`, `tb_sliding_window.sv`, `tb_qkv_transform.sv`, `tb_score_compute.sv`, `tb_softmax_lut.sv`, `tb_attn_output.sv`, `tb_argmax.sv`, `tb_cause_identifier.sv`, `tb_led_buzzer.sv`, `tb_main_sequencer.sv` | 모듈별 기존 단위 시험. 최종 가중치와 최신 RTL로 다시 확인하지 않았으므로 최종 검증 절차에는 포함하지 않는다 |
| `sim/run_all.do` | 기존 스크립트(`top.sv`와 `tb_top.sv`만 컴파일). 이전 라이브러리와 충돌할 수 있어 최종 검증에는 쓰지 않는다 |
| `sim/modelsim.ini` | ModelSim 설정 |

`run_*.do`는 모두 **스크립트 위치(`sim/`)를 기준으로 경로를 계산**하고 시작할 때 이전 시뮬레이션을 닫는다.

## 5. python (모델, 골든 모델, 검증 벡터, PC 인코더)

### 최종 결과물과 파이프라인
| 파일 | 용도 |
| --- | --- |
| `final_model/` | 최종 가중치 `model_fp32.npz`, 내보낸 `weights.mem`, `fc_weights.mem`, `fc_extra.mem`, 표준화 상수 `dataset_meta.json`(센서별 mean, std) |
| `q88_golden.py` | **Q8.8 골든 모델**(NumPy, RTL 비트 단위 재현) + `.mem` 내보내기. 검증의 기준 |
| `make_rtl_vectors.py` | 코어 검증 키트(`kit_fixed`) 생성 -> `run_kit.do`가 사용 |
| `make_e2e_vectors.py` | 통합 검증 키트(`kit_e2e`) 생성(UART 바이트열, 기대 window, 기대 점수, 기대 응답 9바이트) -> `run_e2e.do`가 사용 |
| `sensor_encoder.py` | **PC 쪽 인코더**: 표준화, 클리핑 4σ, 기울기 제한 0.25σ/시점, Q4.4 1바이트, UART 바이트 순서 |
| `phm_diag.py` | PHM 2018 원본 -> 윈도우 cache(`build`), 진단(`analyze`) |
| `make_dataset.py` | cache -> 학습용 합성 이상 데이터셋 |
| `train_attention.py` | 모델 학습(PyTorch), 가중치 내보내기, Q8.8 평가 |

### 분석·실험 도구 (최종 설계 근거, 재현용)
| 파일 | 용도 |
| --- | --- |
| `eval_dataset.py` | 데이터셋의 비선형 기준선(HGB) 평가 |
| `exp_head.py` | 모델 구조 비교(기본 RTL, square, res_square, MLP, 균등 attention 등) |
| `make_dataset_corr.py`, `exp_corr_break.py`, `exp_cross_train.py` | 센서 간 관계 이탈 벤치마크와 학습 데이터셋 교차 평가 |
| `exp_input_encoding.py` | UART 입력 인코딩 비교(Q4.4 채택 근거) |
| `exp_input_range.py`, `exp_demo_sequences.py` | 입력 크기별 넘침 한계, 클리핑·기울기 제한 검증(시연 안전 범위) |
| `extract_signatures_cache.py`, `exp_observable_cv.py`, `check_time_confound.py` | 실제 PHM 고장 event 분석, 팀원 데이터셋 재현 |

스크립트끼리 서로 `import` 하므로 같은 폴더에서 실행한다(파일을 옮기지 않는다). 데이터(`cache/`, `dataset*/`)와 실험 출력(`*_out*/`)은 저장소에 없다.

## 6. 최종 검증 절차

**목적**: 저장소의 RTL과 `mem/`이 골든 모델(`q88_golden.py`)과 비트 단위로 일치하는지 확인한다.

**준비**
1. `mem/`의 세 가중치가 최종 모델과 같은지 확인(PowerShell, 저장소 루트):
```
foreach ($n in 'weights.mem','fc_weights.mem','fc_extra.mem') { if (Compare-Object (Get-Content "mem\$n") (Get-Content "python\final_model\$n")) { "다름: $n" } else { "동일: $n" } }
```
2. 검증 벡터 생성(`python/`에서, 평가 데이터셋 `dataset_t_hard`가 필요하다. 5절의 `phm_diag.py` -> `make_dataset.py`로 만든다):
```
python make_rtl_vectors.py --mode fixed --weights final_model\model_fp32.npz --data dataset_t_hard --mem-dir ..\mem --out kit_fixed
python make_e2e_vectors.py --weights final_model\model_fp32.npz --data dataset_t_hard --mem-dir ..\mem --out kit_e2e
```
`kit_fixed`와 `kit_e2e`는 `.gitignore`로 제외되는 생성 산출물이다.

**실행** (ModelSim 명령창)
```
quit -sim
cd <저장소>/sim
do run_input_scaler.do
do run_kit.do
do run_e2e.do
```

**통과 기준**
| 실행 | 통과 기준 |
| --- | --- |
| `run_input_scaler.do` | `=== input_scaler: ALL PASS ===` |
| `run_kit.do` | 코어 `[RESULT] ALL PASS`(9개 단계 모두 불일치 0 / 216), `fc_layer` PASS, `softmax_lut` PASS |
| `run_e2e.do` | `[RESULT] ALL PASS` — window 77개, 입력 경로, 코어, 출력 경로 모두 PASS |

- 테스트벤치 메시지는 UTF-8 한글이라 ModelSim 창에서 깨져 보일 수 있다. 판정은 영문 `PASS`, `FAIL`, `[RESULT]` 줄로 한다.
- 100MHz로 바꾼 RTL을 검증할 때는 `sim/run_e2e.do`의 `CLK_NS`를 10으로 바꾼다.
- `.mem` 파일은 시뮬레이션 작업 폴더(`kit_fixed`, `kit_e2e`)의 것을 읽고, 합성(Vivado)은 `mem/`의 것을 읽는다. 두 가중치가 같음은 위 준비 1번으로 확인한다.
- FAIL이면 `run_e2e.do`의 표에서 입력 경로 -> 코어 -> 출력 경로 순서로 원인을 좁힌다(앞 단계가 일치하면 뒤 단계의 문제).

## 7. 시연(PC 쪽) 사용 파일
센서 값을 FPGA로 보낼 때는 `python/sensor_encoder.py`와 `python/final_model/dataset_meta.json`을 쓴다.
```python
import json, numpy as np, sensor_encoder as se
meta = json.load(open("final_model/dataset_meta.json", encoding="utf-8"))
z    = se.prepare_stream(raw, np.array(meta["mean"]), np.array(meta["std"]))   # raw: (4, T) 원본 센서 값
data = se.stream_to_bytes(z)                                                   # 시점마다 센서 4바이트
```
- 시점(4바이트) 사이에 **약 2 ms 이상 간격**을 둔다(응답 9바이트 송신 중에 들어온 window는 무시된다).
- 클리핑(4σ)과 기울기 제한(0.25σ/시점)은 FPGA 내부의 산술 넘침을 막는 PC 쪽 보호이므로 우회하지 않는다. FPGA 내부에는 넘침 보호가 없다.

## 8. 범위와 한계
- 모델 성능은 **합성 이상 벤치마크 기준**이다. 실제 PHM 고장 탐지 성능은 주장하지 않는다.
- 검증은 ModelSim 시뮬레이션 수준이다. Vivado 합성(자원, 타이밍)과 보드 시험은 별도 단계다.
- 원인 센서(`cause_index`)는 attention 열합 기반 **참고용 추정**이다.
- 클럭: Basys3의 100MHz 대응(UART 분주 상수)은 이 README의 RTL에 반영되어 있지 않다.
- `docs/design_decisions.md`의 일부 TODO(Softmax 인덱스 슬라이싱 등)는 위 RTL 수정으로 해결되었다.