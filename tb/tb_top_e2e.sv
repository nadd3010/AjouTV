`timescale 1ns / 1ps

// ============================================================================
// tb_top_e2e : top 전체(UART 입력 -> input_scaler -> sliding_window -> 코어 -> UART 응답)를 골든 벡터와 비교한다.
//
//  - make_e2e_vectors.py 가 만든 키트 폴더에서 실행한다 (uart_bytes.mem, e2e_n.mem, e2e_win.mem, e2e_z.mem, e2e_resp.mem 과
//    weights.mem, fc_weights.mem, fc_extra.mem, softmax_*.mem).
//  - 시점 하나(센서 4바이트)를 보낼 때마다 응답 9바이트가 도착할 때까지 기다린 뒤 다음 시점을 보낸다.
//    (응답 송신 중에 들어온 window_ready 는 무시되는 기존 설계의 성질 때문에, 모든 window 가 처리되도록 하려는 것)
//  - window 마다 세 곳을 비교한다:
//      (1) attn_start 시점의 uut.window            vs 기대 window  -> 입력 경로(UART, input_scaler, sliding_window) 점검
//      (2) tx_start 시점의 uut.score_abnormal      vs 기대 점수     -> 코어 점검
//      (3) UART 로 받은 응답 9바이트               vs 기대 응답     -> 출력 경로(argmax, cause_identifier, uart_tx) 점검
//
//  클럭/보레이트: 기본은 현재 저장소(100MHz, 115200). 100MHz 로 바꾸면 컴파일 옵션으로 지정한다.
//      vlog +define+CLK_PERIOD_NS=10 +define+BIT_PERIOD_NS=8680 ...
// ============================================================================
`ifndef CLK_PERIOD_NS
`define CLK_PERIOD_NS 10
`endif
`ifndef BIT_PERIOD_NS
`define BIT_PERIOD_NS 8680
`endif

module tb_top_e2e;

    localparam CLK_PERIOD = `CLK_PERIOD_NS;
    localparam BIT_PERIOD = `BIT_PERIOD_NS;
    localparam MAXT = 512;
    localparam MAXW = MAXT - 3;

    reg        clk;
    reg        rst_n;
    reg        rx;
    wire       tx;
    wire [3:0] led;
    wire       buzzer;

    top uut (
        .clk(clk),
        .rst_n(rst_n),
        .rx(rx),
        .tx(tx),
        .led(led),
        .buzzer(buzzer)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ---------------- 벡터 ----------------
    reg [15:0] nmem  [0:1];
    reg [7:0]  ubytes [0:MAXT*4-1];
    reg [15:0] ewin  [0:MAXW*16-1];
    reg [15:0] ez    [0:MAXW-1];
    reg [7:0]  eresp [0:MAXW*9-1];
    integer T, W;

    // ---------------- 결과 카운터 ----------------
    integer resp_count;       // 응답 9바이트를 모두 받은 window 수
    integer resp_win_bad;     // 응답 바이트가 하나라도 다른 window 수
    integer win_bad;          // attn_start 시점의 window 가 다른 횟수
    integer z_bad;            // tx_start 시점의 score_abnormal 이 다른 횟수
    integer attn_count;       // attn_start 횟수 (= 처리된 window 수)
    integer txs_count;        // tx_start 횟수
    integer shown_win, shown_z, shown_resp;
    reg     done_flag = 0;

    // ---------------- UART ----------------
    task uart_send(input [7:0] data);
        integer i;
        begin
            rx = 1'b0;
            #(BIT_PERIOD);
            for (i = 0; i < 8; i = i + 1) begin
                rx = data[i];
                #(BIT_PERIOD);
            end
            rx = 1'b1;
            #(BIT_PERIOD);
        end
    endtask

    task uart_receive(output [7:0] data);
        integer i;
        begin
            @(negedge tx);
            #(BIT_PERIOD / 2);
            for (i = 0; i < 8; i = i + 1) begin
                #(BIT_PERIOD);
                data[i] = tx;
            end
            #(BIT_PERIOD);
        end
    endtask

    // ---------------- (1) 입력 경로 점검: attn_start 시점의 window ----------------
    always @(posedge uut.attn_start) begin : cap_win
        integer s, t, bad;
        bad = 0;
        if (attn_count < W) begin
            for (s = 0; s < 4; s = s + 1)
                for (t = 0; t < 4; t = t + 1)
                    if (uut.window[s][t] !== ewin[attn_count*16 + s*4 + t]) begin
                        bad = bad + 1;
                        if (shown_win < 6) begin
                            $display("  [INPUT PATH MISMATCH] window %0d [sensor%0d][step%0d]: RTL=%h exp=%h", attn_count, s, t,
                                     uut.window[s][t], ewin[attn_count*16 + s*4 + t]);
                            shown_win = shown_win + 1;
                        end
                    end
            if (bad != 0) win_bad = win_bad + 1;
        end else begin
            $display("  [WARN] more windows processed than expected: attn_start #%0d (expected %0d)", attn_count + 1, W);
        end
        attn_count = attn_count + 1;
    end

    // ---------------- (2) 코어 점검: tx_start 시점의 score_abnormal ----------------
    always @(posedge uut.tx_start) begin : cap_z
        if (txs_count < W) begin
            if (uut.score_abnormal !== ez[txs_count]) begin
                z_bad = z_bad + 1;
                if (shown_z < 6) begin
                    $display("  [CORE MISMATCH] window %0d: score RTL=%h exp=%h", txs_count, uut.score_abnormal, ez[txs_count]);
                    shown_z = shown_z + 1;
                end
            end
        end
        txs_count = txs_count + 1;
    end

    // ---------------- (3) 출력 경로 점검: UART 응답 ----------------
    task run_receiver;
        integer r, j, bad;
        reg [7:0] rb;
        begin
            for (r = 0; r < W; r = r + 1) begin
                bad = 0;
                for (j = 0; j < 9; j = j + 1) begin
                    uart_receive(rb);
                    if (rb !== eresp[r*9 + j]) begin
                        bad = bad + 1;
                        if (shown_resp < 8) begin
                            $display("  [RESPONSE MISMATCH] window %0d byte %0d: RTL=%02X exp=%02X", r, j, rb, eresp[r*9 + j]);
                            shown_resp = shown_resp + 1;
                        end
                    end
                end
                if (bad != 0) resp_win_bad = resp_win_bad + 1;
                resp_count = r + 1;
                if ((r + 1) % 10 == 0) $display("[%0t] window %0d/%0d checked", $time, r + 1, W);
            end
        end
    endtask

    // ---------------- 송신: 시점마다 4바이트, window 가 생기면 응답을 기다림 ----------------
    task run_sender;
        integer k, s, wd;
        begin
            for (k = 0; k < T; k = k + 1) begin
                for (s = 0; s < 4; s = s + 1)
                    uart_send(ubytes[k*4 + s]);
                if (k >= 3) begin
                    wd = 0;
                    while (resp_count < (k - 2) && wd < 400) begin
                        #(BIT_PERIOD);
                        wd = wd + 1;
                    end
                    if (resp_count < (k - 2)) begin
                        $display("[ERROR] no response after step %0d (timeout). responses so far %0d, windows processed %0d, fsm_state=%0d",
                                 k, resp_count, attn_count, uut.fsm_state);
                        k = T;     // 중단
                    end
                end
            end
        end
    endtask

    initial begin
        // 벡터 읽기
        $readmemh("../python/kit_e2e/e2e_n.mem", nmem);
        T = nmem[0];
        W = nmem[1];
        if (T > MAXT || W != T - 3) begin
            $display("[ERROR] bad values in e2e_n.mem (T=%0d, W=%0d, MAXT=%0d)", T, W, MAXT);
            $finish;
        end
        $readmemh("../python/kit_e2euart_bytes.mem", ubytes, 0, T*4 - 1);
        $readmemh("../python/kit_e2ee2e_win.mem",   ewin,   0, W*16 - 1);
        $readmemh("../python/kit_e2ee2e_z.mem",     ez,     0, W - 1);
        $readmemh("../python/kit_e2ee2e_resp.mem",  eresp,  0, W*9 - 1);

        resp_count = 0; resp_win_bad = 0; win_bad = 0; z_bad = 0; attn_count = 0; txs_count = 0;
        shown_win = 0; shown_z = 0; shown_resp = 0; done_flag = 0;

        rst_n = 0;
        rx    = 1'b1;
        #(CLK_PERIOD * 20);
        rst_n = 1;
        #(CLK_PERIOD * 20);
        done_flag = 1;
        $display("=== E2E check start: %0d steps, %0d windows (CLK %0d ns, BIT %0d ns) ===", T, W, CLK_PERIOD, BIT_PERIOD);
        fork
            run_sender;
            run_receiver;
        join

        #(BIT_PERIOD * 5);
        $display("====================================================");
        $display(" Integration result: of %0d windows, processed %0d, responses received %0d", W, attn_count, resp_count);
        $display("  input path (UART->scaler->window)  mismatches %0d  %0s", win_bad,      (win_bad == 0)      ? "PASS" : "FAIL");
        $display("  core (score_abnormal)              mismatches %0d  %0s", z_bad,        (z_bad == 0)        ? "PASS" : "FAIL");
        $display("  output path (UART 9-byte response) mismatches %0d  %0s", resp_win_bad, (resp_win_bad == 0) ? "PASS" : "FAIL");
        if (win_bad == 0 && z_bad == 0 && resp_win_bad == 0 && attn_count == W && resp_count == W)
            $display(" [RESULT] ALL PASS");
        else begin
            $display(" [RESULT] FAIL");
            if (win_bad != 0)       $display("   -> check input path first: input_scaler(Q4.4), uart_rx, sliding_window");
            else if (z_bad != 0)    $display("   -> input path matches. Check the core (if unit/core checks already passed, suspect .mem location or weight mismatch)");
            else                    $display("   -> core matches. Check output path: argmax, cause_identifier, uart_tx");
        end
        $display("====================================================");
`ifdef __ICARUS__
        $finish;
`else
        $stop;
`endif
    end

    // 전체 시간 제한 (window 당 최대 3ms)
    initial begin
        #(64'd3000000 * 100);
        if (!done_flag) begin
            $display("[ERROR] global timeout exceeded");
`ifdef __ICARUS__
            $finish;
`else
            $stop;
`endif
        end
    end

endmodule