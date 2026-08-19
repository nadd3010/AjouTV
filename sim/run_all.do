vlib work
vmap work work

# ===== RTL 소스 컴파일 =====
# 테스트 하고 싶은 파일을 주석 해제

# --- sensor_interface ---
# vlog -sv ../rtl/sensor_interface/uart_rx.sv
# vlog -sv ../rtl/sensor_interface/input_scaler.sv


# --- sliding_window ---
vlog -sv ../rtl/sliding_window/sliding_window.sv

# --- attention_engine ---
=======
# vlog -sv ../rtl/attention_engine/qkv_transform.sv
# vlog -sv ../rtl/attention_engine/score_compute.sv
# vlog -sv ../rtl/attention_engine/softmax_lut.sv
# vlog -sv ../rtl/attention_engine/attn_output.sv

# --- classifier ---
# vlog -sv ../rtl/classifier/fc_layer.sv
# vlog -sv ../rtl/classifier/argmax.sv
# vlog -sv ../rtl/classifier/cause_identifier.sv

# --- output_interface ---
# vlog -sv ../rtl/output_interface/led_controller.sv
# vlog -sv ../rtl/output_interface/buzzer_driver.sv
# vlog -sv ../rtl/output_interface/uart_tx.sv

# --- control_fsm ---
# vlog -sv ../rtl/control_fsm/main_sequencer.sv

# --- top ---
vlog -sv ../rtl/top.sv

# ===== 테스트벤치 컴파일 =====
# vlog -sv ../tb/tb_uart_rx.sv
# vlog -sv ../tb/tb_input_scaler.sv
# vlog -sv ../tb/tb_sliding_window.sv
# vlog -sv ../tb/tb_qkv_transform.sv
# vlog -sv ../tb/tb_score_compute.sv
# vlog -sv ../tb/tb_softmax_lut.sv
# vlog -sv ../tb/tb_attn_output.sv
# vlog -sv ../tb/tb_fc_layer.sv
# vlog -sv ../tb/tb_argmax.sv
# vlog -sv ../tb/tb_led_buzzer.sv
# vlog -sv ../tb/tb_buzzer_driver.sv
# vlog -sv ../tb/tb_uart_tx.sv
# vlog -sv ../tb/tb_main_sequencer.sv
vlog -sv ../tb/tb_top.sv


# ===== 시뮬레이션 실행 =====
# vsim work.tb_uart_rx
# vsim work.tb_input_scaler
# vsim work.tb_sliding_window
# vsim work.tb_qkv_transform
# vsim work.tb_score_compute
# vsim work.tb_softmax_lut
# vsim work.tb_attn_output
# vsim work.tb_fc_layer
# vsim work.tb_argmax
# vsim work.tb_cause_identifier
# vsim work.tb_led_buzzer
# vsim work.tb_uart_tx
# vsim work.tb_main_sequencer
vsim work.tb_top

# ===== 파형 추가 =====
add wave -radix hex sim:/*

# qkv_transform 할 때 
# add wave -radix hex sim:/tb_qkv_transform/q
# add wave -radix hex sim:/tb_qkv_transform/k
# add wave -radix hex sim:/tb_qkv_transform/v

# ===== 실행 시간 =====
run -all
