# ModelSim 실행 스크립트: top 전체(UART -> 슬라이딩 윈도우 -> 코어 -> 출력) 통합 시뮬레이션 (tb_top)
#
# 기존 sim/run_all.do 는 top.sv 와 tb_top.sv 만 다시 컴파일하고 나머지 RTL 은 주석 처리되어 있어서,
# sim/work 라이브러리에 남아 있는 '예전에 컴파일된 모듈'(포트가 다른 옛 fc_layer, uart_tx 등)을 불러와 연결 오류가 난다.
# 이 스크립트는 모든 RTL 을 새로 컴파일한다. 기존 run_all.do 는 건드리지 않는다.
#
# 작업 폴더를 키트 폴더로 옮겨 키트의 .mem(weights, fc_weights, fc_extra, softmax_exp_rom, softmax_inv_rom)을 읽는다.
# (RTL 이 $readmemh 에 파일 이름만 쓰므로, sim 폴더에 있는 예전 .mem 과 섞이지 않게 하려는 것)

set _s [info script]
if {$_s eq ""} { set SIM_DIR [pwd] } else { set SIM_DIR [file dirname [file normalize $_s]] }
set REPO [file normalize "$SIM_DIR/.."]
set KIT  "$REPO/python/kit_fixed"

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

vlog -sv {*}$SRC $REPO/tb/tb_top.sv
vsim work.tb_top
run -all
