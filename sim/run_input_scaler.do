# ModelSim: Q4.4 입력 스케일러 단위 시험 (tb_input_scaler)
# 사용법:  quit -sim  /  cd <AjouTV>/sim  /  do run_input_scaler.do   (현재 폴더가 sim 이어야 한다)
set _s [info script]
if {$_s eq ""} { set SIM_DIR [pwd] } else { set SIM_DIR [file dirname [file normalize $_s]] }
set REPO [file normalize "$SIM_DIR/.."]
if {![file exists "$REPO/rtl/top.sv"]} { echo "(ERROR) repo not found from $SIM_DIR. In ModelSim run: cd <AjouTV>/sim   then run this script again"; return }
catch {quit -sim}
if {[file exists $SIM_DIR/work_scaler]} { vdel -lib $SIM_DIR/work_scaler -all }
vlib $SIM_DIR/work_scaler
vmap work $SIM_DIR/work_scaler
vlog -sv $REPO/rtl/sensor_interface/input_scaler.sv $REPO/tb/tb_input_scaler.sv
vsim work.tb_input_scaler
run -all
