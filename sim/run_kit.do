# ModelSim 실행 스크립트 (RTL 수정 키트 검증)
#
# 사용법
#   1) 경로는 이 스크립트 위치(AjouTV/sim)를 기준으로 자동 계산한다. 키트는 python/kit_fixed (make_rtl_vectors.py 결과).
#   2) 수정한 rtl/, tb/ 파일이 저장소에 반영된 것을 확인한 뒤 ModelSim 에서 실행한다.
#        quit -sim / cd <AjouTV>/sim / do run_kit.do
#   (저장소의 기존 sim/run_all.do 는 건드리지 않는다. 이 파일은 별도로 추가해서 쓴다.)
#   수정 전 RTL(main)을 확인하려면 PATCHED 정의를 빈 문자열로 바꾸고, 키트도 --mode orig 로 만든 것을 쓴다.

set _s [info script]
if {$_s eq ""} { set SIM_DIR [pwd] } else { set SIM_DIR [file dirname [file normalize $_s]] }
set REPO [file normalize "$SIM_DIR/.."]
set KIT  "$REPO/python/kit_fixed"
set DEFS "+define+PATCHED"

# 작업 폴더를 키트 폴더로 바꾼다 (모듈이 .mem 을 파일 이름만으로 읽기 때문). 저장소 sim 폴더의 예전 .mem 과 섞이지 않는다.
# 이전 시뮬레이션이 열려 있으면 폴더를 바꿀 수 없으므로 먼저 닫는다 ($stop 으로 멈춘 뒤에도 시뮬레이션은 열려 있다)
catch {quit -sim}
cd $KIT

if {[file exists work]} { vdel -lib work -all }
vlib work
vmap work work

set SRC [list \
  $REPO/rtl/control_fsm/main_sequencer.sv \
  $REPO/rtl/attention_engine/qkv_transform.sv \
  $REPO/rtl/attention_engine/score_compute.sv \
  $REPO/rtl/attention_engine/softmax_lut.sv \
  $REPO/rtl/attention_engine/attn_output.sv \
  $REPO/rtl/classifier/fc_layer.sv \
  $REPO/rtl/classifier/argmax.sv \
  $REPO/rtl/classifier/cause_identifier.sv ]

# 1) 코어 데이터패스 전체 (단계별 비교 + 지연)
vlog -sv $DEFS {*}$SRC $REPO/tb/tb_core_e2e.sv
vsim work.tb_core_e2e
run -all
quit -sim

# 2) fc_layer 단위 테스트
vlog -sv $REPO/rtl/classifier/fc_layer.sv $REPO/tb/tb_fc_layer.sv
vsim work.tb_fc_layer
run -all
quit -sim

# 3) softmax_lut 단위 테스트
vlog -sv $REPO/rtl/attention_engine/softmax_lut.sv $REPO/tb/tb_softmax_lut_vec.sv
vsim work.tb_softmax_lut_vec
run -all
quit -sim

echo "=== done: transcript 의 RESULT, 단위 테스트 PASS 줄을 확인하세요 ==="
