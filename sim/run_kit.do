# ModelSim 실행 스크립트 (최종 검증: 코어 데이터패스 + fc_layer + softmax_lut)
#
# 사용법:  quit -sim  /  cd <AjouTV>/sim  /  do run_kit.do
#   1) 경로는 이 스크립트 위치(AjouTV/sim)를 기준으로 자동 계산한다.
#   2) 테스트 입력과 기대값은 키트(python/kit_fixed, make_rtl_vectors.py 결과)에서 읽는다.
#   3) 모델 .mem 5개(weights, fc_weights, fc_extra, softmax_exp_rom, softmax_inv_rom)는 저장소 mem/ 의 파일을
#      키트 폴더로 덮어써서 사용한다. 즉 검증에 쓰이는 .mem 은 항상 mem/ 의 파일이다.
#      (RTL 은 파일 이름만으로 .mem 을 읽는다. Vivado 도 같은 RTL 을 쓰므로 RTL 은 고치지 않고, 시뮬레이션 쪽에서 위치를 맞춘다.)
#   (저장소의 기존 sim/run_all.do 는 건드리지 않는다.)
#   수정 전 RTL(main)을 확인하려면 PATCHED 정의를 빈 문자열로 바꾸고, 키트도 --mode orig 로 만든 것을 쓴다.

set _s [info script]
if {$_s eq ""} { set SIM_DIR [pwd] } else { set SIM_DIR [file dirname [file normalize $_s]] }
set REPO   [file normalize "$SIM_DIR/.."]
set KIT    "$REPO/python/kit_fixed"
set MEMDIR "$REPO/mem"
set DEFS   "+define+PATCHED"

set MEMFILES [list weights.mem fc_weights.mem fc_extra.mem softmax_exp_rom.mem softmax_inv_rom.mem]
foreach f $MEMFILES {
    if {![file exists "$MEMDIR/$f"]} { echo "(ERROR) not found: $MEMDIR/$f"; return }
}
if {![file exists "$KIT/vec_n.mem"]} {
    echo "(ERROR) kit folder not found: $KIT  (run make_rtl_vectors.py in python/ first)"
    return
}

# 이전 시뮬레이션이 열려 있으면 폴더를 바꿀 수 없으므로 먼저 닫는다 ($stop 으로 멈춘 뒤에도 시뮬레이션은 열려 있다)
catch {quit -sim}
cd $KIT

# 모델 .mem 을 저장소 mem/ 의 파일로 교체 (이후 시뮬레이션은 모두 이 파일을 읽는다)
foreach f $MEMFILES { file copy -force "$MEMDIR/$f" "$KIT/$f" }
echo "(MEM) model .mem files taken from $MEMDIR"

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

echo "=== done: check the RESULT line and the unit test PASS lines in the transcript ==="
