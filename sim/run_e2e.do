# ModelSim: UART 입력 -> 코어 -> UART 응답 전체를 골든 벡터와 비교 (tb_top_e2e)
# 사용법:  quit -sim  /  cd <AjouTV>/sim  /  do run_e2e.do   (ModelSim 은 do 안에서 스크립트 위치를 알려주지 않아, 현재 폴더가 sim 이어야 한다)
# 준비: python/ 에서 make_e2e_vectors.py 로 python/kit_e2e 를 만든다. 그 폴더가 시뮬레이션 작업 폴더가 된다.
#   테스트 입력과 기대값은 kit_e2e 에서 읽고, 모델 .mem 5개는 저장소 mem/ 의 파일을 kit_e2e 로 덮어써서 사용한다.
#   (RTL 은 파일 이름만으로 .mem 을 읽는다. Vivado 도 같은 RTL 을 쓰므로 RTL 은 고치지 않는다.)
# 클럭: 현재 main RTL 은 100MHz 기준(CLKS_PER_BIT = 868)이라 CLK_NS 는 10. 50MHz RTL 을 검증할 때만 20 으로 바꾼다.
set _s [info script]
if {$_s eq ""} { set SIM_DIR [pwd] } else { set SIM_DIR [file dirname [file normalize $_s]] }
set REPO   [file normalize "$SIM_DIR/.."]
set KIT    "$REPO/python/kit_e2e"
set MEMDIR "$REPO/mem"
set TBFILE "$REPO/tb/tb_top_e2e.sv"
if {![file exists "$REPO/rtl/top.sv"]} { echo "(ERROR) repo not found from $SIM_DIR. In ModelSim run: cd <AjouTV>/sim   then run this script again"; return }
if {![file exists $KIT/e2e_n.mem]}   { echo "(ERROR) kit folder not found: $KIT  (run make_e2e_vectors.py in python/ first)"; return }
set MEMFILES [list weights.mem fc_weights.mem fc_extra.mem softmax_exp_rom.mem softmax_inv_rom.mem]
foreach f $MEMFILES {
    if {![file exists "$MEMDIR/$f"]} { echo "(ERROR) not found: $MEMDIR/$f"; return }
}
set CLK_NS 10
set BIT_NS 8680

catch {quit -sim}
cd $KIT

# 모델 .mem 을 저장소 mem/ 의 파일로 교체
foreach f $MEMFILES { file copy -force "$MEMDIR/$f" "$KIT/$f" }
echo "(MEM) model .mem files taken from $MEMDIR"

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
