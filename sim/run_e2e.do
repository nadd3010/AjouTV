# ============================================================================
# run_e2e.do
#
# 기존 tb_top_e2e.sv 및 RTL을 수정하지 않고 E2E 테스트 실행
#
# ModelSim Transcript에서:
#   quit -sim
#   cd "C:/Users/LG/Desktop/AjouTV/sim"
#   do run_e2e.do
#
# 경로 기준:
#   RTL 모델 파일: ../mem/*.mem
#   테스트 개수:   ../python/kit_e2e/e2e_n.mem
#
# 테스트벤치에 슬래시가 빠진 경로가 있으므로,
# 해당 경로에서 읽을 수 있는 벡터 복사본을 준비한다.
# ============================================================================

# 명령 실행 중 오류가 나면 스크립트 중단
onerror {abort}

# 반드시 AjouTV/sim에서 실행
set SIM_DIR [file normalize [pwd]]
set REPO    [file normalize [file join $SIM_DIR ".."]]
set KIT     [file join $REPO "python" "kit_e2e"]
set MEMDIR  [file join $REPO "mem"]
set TBFILE  [file join $REPO "tb" "tb_top_e2e.sv"]

# 100 MHz, UART 868 clocks/bit
set CLK_NS 10
set BIT_NS 8680

# ============================================================================
# 1. 실행 위치 및 테스트벤치 확인
# ============================================================================

if {![file isdirectory [file join $REPO "rtl"]]} {
    error "Wrong working directory. Run from AjouTV/sim. Current: $SIM_DIR"
}

if {![file readable [file join $REPO "rtl" "top.sv"]]} {
    error "Cannot read top.sv under: $REPO"
}

if {![file readable $TBFILE]} {
    error "Cannot read testbench: $TBFILE"
}

echo "(SIM)  $SIM_DIR"
echo "(REPO) $REPO"
echo "(KIT)  $KIT"

# ============================================================================
# 2. 모델 파일 확인
# ============================================================================

set MEMFILES {
    weights.mem
    fc_weights.mem
    fc_extra.mem
    softmax_exp_rom.mem
    softmax_inv_rom.mem
}

foreach f $MEMFILES {
    set model_path [file join $MEMDIR $f]

    if {![file readable $model_path]} {
        error "Cannot read model file: $model_path"
    }
}

# ============================================================================
# 3. 테스트 벡터 파일 확인
# ============================================================================

set VECTOR_FILES {
    e2e_n.mem
    uart_bytes.mem
    e2e_win.mem
    e2e_z.mem
    e2e_resp.mem
}

foreach f $VECTOR_FILES {
    set vector_path [file join $KIT $f]

    if {![file readable $vector_path]} {
        error "Cannot read vector file: $vector_path"
    }
}

# ============================================================================
# 4. 컴파일할 RTL 목록 및 파일 확인
# ============================================================================

set SRC [list \
    [file join $REPO "rtl/sensor_interface/uart_rx.sv"] \
    [file join $REPO "rtl/sensor_interface/input_scaler.sv"] \
    [file join $REPO "rtl/sliding_window/sliding_window.sv"] \
    [file join $REPO "rtl/attention_engine/qkv_transform.sv"] \
    [file join $REPO "rtl/attention_engine/score_compute.sv"] \
    [file join $REPO "rtl/attention_engine/softmax_lut.sv"] \
    [file join $REPO "rtl/attention_engine/attn_output.sv"] \
    [file join $REPO "rtl/classifier/fc_layer.sv"] \
    [file join $REPO "rtl/classifier/argmax.sv"] \
    [file join $REPO "rtl/classifier/cause_identifier.sv"] \
    [file join $REPO "rtl/output_interface/led_controller.sv"] \
    [file join $REPO "rtl/output_interface/buzzer_driver.sv"] \
    [file join $REPO "rtl/output_interface/uart_tx.sv"] \
    [file join $REPO "rtl/control_fsm/main_sequencer.sv"] \
    [file join $REPO "rtl/top.sv"] \
]

foreach f $SRC {
    if {![file readable $f]} {
        error "Cannot read RTL source: $f"
    }
}

# ============================================================================
# 5. 기존 시뮬레이션 종료 및 작업 폴더 유지
# ============================================================================

catch {quit -sim}

# 중요: KIT로 이동하지 않는다.
# 기존 RTL과 테스트벤치의 상대 경로는 sim 폴더 기준이다.
cd $SIM_DIR

# ============================================================================
# 6. 기존 테스트벤치가 찾는 이름으로 벡터 복사본 준비
#
# 예:
#   원본: python/kit_e2e/uart_bytes.mem
#   복사: python/kit_e2euart_bytes.mem
#
# 원본 벡터는 유지하며, 재실행할 때 복사본을 최신 내용으로 갱신한다.
# ============================================================================

set VECTOR_ALIASES {
    uart_bytes.mem kit_e2euart_bytes.mem
    e2e_win.mem    kit_e2ee2e_win.mem
    e2e_z.mem      kit_e2ee2e_z.mem
    e2e_resp.mem   kit_e2ee2e_resp.mem
}

foreach {src_name dst_name} $VECTOR_ALIASES {
    set src [file join $KIT $src_name]
    set dst [file join $REPO "python" $dst_name]

    file copy -force $src $dst
    echo "(VECTOR) $src -> $dst"
}

# ============================================================================
# 7. 실제 readmem 경로 확인
# ============================================================================

set REQUIRED_FILES {
    ../mem/weights.mem
    ../mem/fc_weights.mem
    ../mem/fc_extra.mem
    ../mem/softmax_exp_rom.mem
    ../mem/softmax_inv_rom.mem
    ../python/kit_e2e/e2e_n.mem
    ../python/kit_e2euart_bytes.mem
    ../python/kit_e2ee2e_win.mem
    ../python/kit_e2ee2e_z.mem
    ../python/kit_e2ee2e_resp.mem
}

foreach f $REQUIRED_FILES {
    if {![file readable $f]} {
        error "Cannot read simulation data: [file normalize $f]"
    }
}

echo "(OK) All required .mem files are readable."
echo "(MEM) RTL will use model files from $MEMDIR"

# ============================================================================
# 8. work 라이브러리 준비
# ============================================================================

set WORK_DIR [file join $SIM_DIR "work"]

if {![file exists $WORK_DIR]} {
    vlib $WORK_DIR
}

vmap work $WORK_DIR

# ============================================================================
# 9. RTL 및 기존 테스트벤치 컴파일
# ============================================================================

echo "(COMPILE) RTL and tb_top_e2e.sv"

vlog -sv \
    +define+CLK_PERIOD_NS=$CLK_NS \
    +define+BIT_PERIOD_NS=$BIT_NS \
    {*}$SRC \
    $TBFILE

# ============================================================================
# 10. 시뮬레이션 실행
# ============================================================================

echo "(RUN) tb_top_e2e"

vsim work.tb_top_e2e
run -all