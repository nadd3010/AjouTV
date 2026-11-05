# ModelSim: UART 입력 -> 코어 -> UART 응답 전체를 골든 벡터와 비교 (tb_top_e2e)
# 사용법:  quit -sim  /  cd <AjouTV>/sim  /  do run_e2e.do
# 준비: python/ 에서 make_e2e_vectors.py 로 python/kit_e2e 를 만든다. 그 폴더가 시뮬레이션 작업 폴더가 된다.
# 클럭: 기본 50MHz(CLK_NS 20). 100MHz 로 바꾼 RTL 을 검증할 때는 CLK_NS 를 10 으로 바꾼다.
set _s [info script]
if {$_s eq ""} { set SIM_DIR [pwd] } else { set SIM_DIR [file dirname [file normalize $_s]] }
set REPO   [file normalize "$SIM_DIR/.."]
set KIT    "$REPO/python/kit_e2e"
set TBFILE "$REPO/tb/tb_top_e2e.sv"
if {![file exists $REPO/rtl/top.sv]} { echo "[오류] 저장소 경로를 찾지 못했습니다: $REPO"; return }
if {![file exists $KIT/e2e_n.mem]}   { echo "[오류] 키트 폴더가 없습니다: $KIT  (python/ 에서 make_e2e_vectors.py 를 먼저 실행)"; return }
set CLK_NS 20
set BIT_NS 8680

catch {quit -sim}
cd $KIT
if {[file exists work]} { vdel -lib work -all }
vlib work
vmap work work

set SRC [list \
  $REPO/rtl/sensor_interface/uart_rx.sv \
  $REPO/rtl/sensor_interface/input_scaler.sv \
  $REPO/rtl/sliding_window/sliding_window.sv \
  $REPO/rtl/attention_engine/qkv_transform.sv \
  $REPO/rtl/attention_engine/score_compute.sv \
  $REPO/rtl/attention_engine/softmax_lut.sv \
  $REPO/rtl/attention_engine/attn_output.sv \
  $REPO/rtl/classifier/fc_layer.sv \
  $REPO/rtl/classifier/argmax.sv \
  $REPO/rtl/classifier/cause_identifier.sv \
  $REPO/rtl/output_interface/led_controller.sv \
  $REPO/rtl/output_interface/buzzer_driver.sv \
  $REPO/rtl/output_interface/uart_tx.sv \
  $REPO/rtl/control_fsm/main_sequencer.sv \
  $REPO/rtl/top.sv ]

vlog -sv +define+CLK_PERIOD_NS=$CLK_NS +define+BIT_PERIOD_NS=$BIT_NS {*}$SRC $TBFILE
vsim work.tb_top_e2e
run -all
