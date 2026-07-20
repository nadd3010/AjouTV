# ===== RTL 소스 컴파일 =====
# 테스트 하고 싶은 파일을 주석 해제

# --- sensor_interface ---
vlog ../rtl/sensor_interface/uart_rx.v
# vlog ../rtl/sensor_interface/input_scaler.v

# --- sliding_window ---
# vlog ../rtl/sliding_window/sliding_window.v

# --- attention_engine ---
# vlog ../rtl/attention_engine/qkv_transform.v
# vlog ../rtl/attention_engine/score_compute.v
# vlog ../rtl/attention_engine/softmax_lut.v
# vlog ../rtl/attention_engine/attn_output.v

# --- classifier ---
# vlog ../rtl/classifier/fc_layer.v
# vlog ../rtl/classifier/argmax.v
# vlog ../rtl/classifier/cause_identifier.v

# --- output_interface ---
# vlog ../rtl/output_interface/led_controller.v
# vlog ../rtl/output_interface/buzzer_driver.v
# vlog ../rtl/output_interface/uart_tx.v

# --- control_fsm ---
# vlog ../rtl/control_fsm/main_sequencer.v

# --- top ---
# vlog ../rtl/top.v

# ===== 테스트벤치 컴파일 =====
vlog ../tb/tb_uart_rx.v
# vlog ../tb/tb_input_scaler.v
# vlog ../tb/tb_sliding_window.v
# vlog ../tb/tb_qkv_transform.v
# vlog ../tb/tb_score_compute.v
# vlog ../tb/tb_softmax_lut.v
# vlog ../tb/tb_attn_output.v
# vlog ../tb/tb_fc_layer.v
# vlog ../tb/tb_argmax.v
# vlog ../tb/tb_cause_identifier.v
# vlog ../tb/tb_led_controller.v
# vlog ../tb/tb_buzzer_driver.v
# vlog ../tb/tb_uart_tx.v
# vlog ../tb/tb_main_sequencer.v
# vlog ../tb/tb_top.v

# ===== 시뮬레이션 실행 =====
vsim work.tb_uart_rx
# vsim work.tb_input_scaler
# vsim work.tb_sliding_window
# vsim work.tb_qkv_transform
# vsim work.tb_score_compute
# vsim work.tb_softmax_lut
# vsim work.tb_attn_output
# vsim work.tb_fc_layer
# vsim work.tb_argmax
# vsim work.tb_cause_identifier
# vsim work.tb_led_controller
# vsim work.tb_buzzer_driver
# vsim work.tb_uart_tx
# vsim work.tb_main_sequencer
# vsim work.tb_top

# ===== 파형 추가 =====
add wave -radix hex sim:/*

# ===== 실행 시간 =====
run 1ms
# run -all