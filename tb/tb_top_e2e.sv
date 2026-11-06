
`timescale 1ns / 1ps

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

    reg clk;
    reg rst_n;
    reg rx;

    wire tx;
    wire [3:0] led;
    wire buzzer;

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

    // =====================================================
    // 1. Golden Model Vectors
    // =====================================================
    reg [15:0] nmem [0:1];
    reg [7:0] ubytes [0:MAXT*4-1];
    reg [15:0] ewin [0:MAXW*16-1];
    reg [15:0] ez [0:MAXW-1];
    reg [7:0] eresp [0:MAXW*9-1];

    integer T, W;

    // =====================================================
    // 2. Verification Counters
    // =====================================================
    integer resp_count;
    integer resp_win_bad;
    integer win_bad;
    integer z_bad;
    integer attn_count;
    integer txs_count;
    integer shown_win, shown_z, shown_resp;

    reg finished_flag = 0;

    // =====================================================
    // 3. Performance Measurement
    //
    // 마지막 센서 시점의 첫 UART Start Bit
    // window_ready
    // classify_done
    // 마지막 UART 응답 Stop Bit 종료
    // =====================================================
    realtime sample_start_ns [0:MAXW-1];
    realtime window_ready_ns [0:MAXW-1];
    realtime classify_done_ns [0:MAXW-1];
    realtime response_done_ns [0:MAXW-1];

    integer ready_count;
    integer classify_count;

    real infer_sum_ns, infer_min_ns, infer_max_ns;
    real resp_sum_ns, resp_min_ns, resp_max_ns;
    real e2e_sum_ns, e2e_min_ns, e2e_max_ns;

    real infer_ns, response_ns, e2e_ns;
    real completion_interval_ns;
    real system_throughput;

    // =====================================================
    // 4. UART Send
    // =====================================================
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

    // =====================================================
    // 5. UART Receive
    //
    // 마지막 대기 후 Stop Bit 중앙에 도달
    // =====================================================
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

    // =====================================================
    // 6. Window Ready Timestamp
    // =====================================================
    always @(posedge uut.window_ready) begin
        if (ready_count < MAXW) begin
            window_ready_ns[ready_count] = $realtime;
        end
        ready_count = ready_count + 1;
    end

    // =====================================================
    // 7. Classification Done Timestamp
    // =====================================================
    always @(posedge uut.classify_done) begin
        if (classify_count < MAXW) begin
            classify_done_ns[classify_count] = $realtime;
        end
        classify_count = classify_count + 1;
    end

    // =====================================================
    // 8. Input Path Golden Check
    // =====================================================
    always @(posedge uut.attn_start) begin : cap_win
        integer s, t, bad;

        bad = 0;

        if (attn_count < W) begin
            for (s = 0; s < 4; s = s + 1) begin
                for (t = 0; t < 4; t = t + 1) begin

                    if (uut.window[s][t] !==
                        ewin[attn_count*16 + s*4 + t]) begin

                        bad = bad + 1;

                        if (shown_win < 6) begin
                            $display(
                                "[INPUT MISMATCH] window %0d [%0d][%0d] RTL=%h exp=%h",
                                attn_count, s, t,
                                uut.window[s][t],
                                ewin[attn_count*16+s*4+t]
                            );
                            shown_win = shown_win + 1;
                        end
                    end
                end
            end

            if (bad != 0)
                win_bad = win_bad + 1;

        end else begin
            $display("[WARN] Extra window detected");
        end

        attn_count = attn_count + 1;
    end

    // =====================================================
    // 9. Core Golden Check
    // =====================================================
    always @(posedge uut.tx_start) begin : cap_z

        if (txs_count < W) begin

            if (uut.score_abnormal !== ez[txs_count]) begin
                z_bad = z_bad + 1;

                if (shown_z < 6) begin
                    $display(
                        "[CORE MISMATCH] window %0d RTL=%h exp=%h",
                        txs_count,
                        uut.score_abnormal,
                        ez[txs_count]
                    );
                    shown_z = shown_z + 1;
                end
            end
        end

        txs_count = txs_count + 1;
    end

    // =====================================================
    // 10. UART Response Receiver
    // =====================================================
    task run_receiver;

        integer r, j, bad;
        reg [7:0] rb;

        begin
            for (r = 0; r < W; r = r + 1) begin

                bad = 0;

                for (j = 0; j < 9; j = j + 1) begin

                    uart_receive(rb);

                    if (rb !== eresp[r*9+j]) begin
                        bad = bad + 1;

                        if (shown_resp < 8) begin
                            $display(
                                "[RESPONSE MISMATCH] window %0d byte %0d RTL=%02X exp=%02X",
                                r, j, rb, eresp[r*9+j]
                            );
                            shown_resp = shown_resp + 1;
                        end
                    end
                end

                // uart_receive은 Stop Bit 중앙에서 반환함.
                // 마지막 바이트의 Stop Bit 끝까지 기다려
                // 응답 프레임 완료 시각을 기록한다.
                #(BIT_PERIOD / 2);

                response_done_ns[r] = $realtime;

                if (bad != 0)
                    resp_win_bad = resp_win_bad + 1;

                resp_count = r + 1;

                if ((r + 1) % 10 == 0)
                    $display(
                        "[%0t] window %0d/%0d checked",
                        $time, r+1, W
                    );
            end
        end
    endtask

    // =====================================================
    // 11. UART Input Sender
    //
    // 4바이트 최신 센서 시점의 송신 시작 시각 기록
    // =====================================================
    task run_sender;

        integer k, s, wd;

        begin
            for (k = 0; k < T; k = k + 1) begin

                // 네 번째 시점부터 새 윈도우가 발생
                if (k >= 3) begin
                    sample_start_ns[k-3] = $realtime;
                end

                for (s = 0; s < 4; s = s + 1) begin
                    uart_send(ubytes[k*4+s]);
                end

                // 응답이 끝나기 전 다음 시점 전송 금지
                if (k >= 3) begin

                    wd = 0;

                    while (resp_count < (k-2) && wd < 400) begin
                        #(BIT_PERIOD);
                        wd = wd + 1;
                    end

                    if (resp_count < (k-2)) begin
                        $fatal(
                            1,
                            "[ERROR] Timeout after step %0d",
                            k
                        );
                    end
                end
            end
        end
    endtask

    // =====================================================
    // 12. Main Test
    // =====================================================
    initial begin

        // ---------------------------------------------
        // Load Golden Vectors
        // ---------------------------------------------
        $readmemh("../python/kit_e2e/e2e_n.mem", nmem);

        T = nmem[0];
        W = nmem[1];

        if (T > MAXT || W <= 0 || W != T-3) begin
            $fatal(1, "[ERROR] Invalid vector count");
        end

        $readmemh("../python/kit_e2e/uart_bytes.mem",
                  ubytes, 0, T*4-1);
        $readmemh("../python/kit_e2e/e2e_win.mem",
                  ewin, 0, W*16-1);
        $readmemh("../python/kit_e2e/e2e_z.mem",
                  ez, 0, W-1);
        $readmemh("../python/kit_e2e/e2e_resp.mem",
                  eresp, 0, W*9-1);

        resp_count = 0;
        resp_win_bad = 0;
        win_bad = 0;
        z_bad = 0;
        attn_count = 0;
        txs_count = 0;
        ready_count = 0;
        classify_count = 0;

        shown_win = 0;
        shown_z = 0;
        shown_resp = 0;

        finished_flag = 0;

        infer_sum_ns = 0.0;
        resp_sum_ns = 0.0;
        e2e_sum_ns = 0.0;

        infer_min_ns = 1.0e30;
        resp_min_ns = 1.0e30;
        e2e_min_ns = 1.0e30;

        infer_max_ns = 0.0;
        resp_max_ns = 0.0;
        e2e_max_ns = 0.0;

        for (int r = 0; r < MAXW; r++) begin
            sample_start_ns[r] = 0;
            window_ready_ns[r] = 0;
            classify_done_ns[r] = 0;
            response_done_ns[r] = 0;
        end

        // ---------------------------------------------
        // Reset
        // ---------------------------------------------
        rst_n = 0;
        rx = 1'b1;

        #(CLK_PERIOD * 20);
        rst_n = 1;
        #(CLK_PERIOD * 20);

        $display(
            "=== E2E check start: %0d steps, %0d windows ===",
            T, W
        );

        // ---------------------------------------------
        // Run sender / receiver
        // ---------------------------------------------
        fork
            run_sender;
            run_receiver;
        join

        #(BIT_PERIOD * 5);

        // ---------------------------------------------
        // Functional Verification Summary
        // ---------------------------------------------
        $display("");
        $display("========================================");
        $display(" INTEGRATION FUNCTIONAL CHECK");
        $display("========================================");

        $display(" Expected windows : %0d", W);
        $display(" Processed windows: %0d", attn_count);
        $display(" UART responses   : %0d", resp_count);

        $display(" Input mismatches : %0d", win_bad);
        $display(" Core mismatches  : %0d", z_bad);
        $display(" UART mismatches  : %0d", resp_win_bad);

        if (win_bad == 0 &&
            z_bad == 0 &&
            resp_win_bad == 0 &&
            attn_count == W &&
            txs_count == W &&
            resp_count == W &&
            ready_count == W &&
            classify_count == W)
            $display(" [RESULT] ALL PASS");
        else
            $fatal(1, "[RESULT] FUNCTIONAL FAIL");

        // ---------------------------------------------
        // Performance Statistics
        // ---------------------------------------------
        for (int r = 0; r < W; r++) begin

            infer_ns =
                classify_done_ns[r] - window_ready_ns[r];

            response_ns =
                response_done_ns[r] - window_ready_ns[r];

            e2e_ns =
                response_done_ns[r] - sample_start_ns[r];

            if (infer_ns < 0 || response_ns < 0 ||
                e2e_ns < 0)
                $fatal(1, "[ERROR] Invalid timestamps");

            infer_sum_ns = infer_sum_ns + infer_ns;
            resp_sum_ns = resp_sum_ns + response_ns;
            e2e_sum_ns = e2e_sum_ns + e2e_ns;

            if (infer_ns < infer_min_ns)
                infer_min_ns = infer_ns;

            if (infer_ns > infer_max_ns)
                infer_max_ns = infer_ns;

            if (response_ns < resp_min_ns)
                resp_min_ns = response_ns;

            if (response_ns > resp_max_ns)
                resp_max_ns = response_ns;

            if (e2e_ns < e2e_min_ns)
                e2e_min_ns = e2e_ns;

            if (e2e_ns > e2e_max_ns)
                e2e_max_ns = e2e_ns;
        end

        $display("");
        $display("========================================");
        $display(" SYSTEM PERFORMANCE @ 100MHz");
        $display("========================================");

        $display("[INFERENCE LATENCY]");
        $display(" Window ready -> Classification done");
        $display(" Min : %.3f us", infer_min_ns / 1000.0);
        $display(" Avg : %.3f us", infer_sum_ns / W / 1000.0);
        $display(" Max : %.3f us", infer_max_ns / 1000.0);

        $display("");
        $display("[RESPONSE LATENCY]");
        $display(" Window ready -> UART response complete");
        $display(" Min : %.3f us", resp_min_ns / 1000.0);
        $display(" Avg : %.3f us", resp_sum_ns / W / 1000.0);
        $display(" Max : %.3f us", resp_max_ns / 1000.0);

        $display("");
        $display("[END-TO-END LATENCY]");
        $display(" Last sample UART start -> response complete");
        $display(" Min : %.3f us", e2e_min_ns / 1000.0);
        $display(" Avg : %.3f us", e2e_sum_ns / W / 1000.0);
        $display(" Max : %.3f us", e2e_max_ns / 1000.0);

        // ---------------------------------------------
        // Sustained Throughput
        // ---------------------------------------------
        if (W > 1) begin

            completion_interval_ns =
                (response_done_ns[W-1] -
                 response_done_ns[0]) / (W-1);

            system_throughput =
                1.0e9 / completion_interval_ns;

            $display("");
            $display("[SYSTEM THROUGHPUT]");
            $display(" Completed responses : %0d", W);
            $display(
                " Avg response interval : %.3f us",
                completion_interval_ns / 1000.0
            );
            $display(
                " Sustained throughput  : %.2f responses/sec",
                system_throughput
            );

            if (system_throughput >= 2000.0)
                $display(" [TARGET] 2000 responses/sec MET");
            else
                $display(
                    " [TARGET] 2000 responses/sec NOT MET in this UART scenario"
                );
        end

        $display("========================================");

        finished_flag = 1;

`ifdef __ICARUS__
        $finish;
`else
        $stop;
`endif
    end

    // =====================================================
    // 13. Global Timeout
    // =====================================================
    initial begin
        #(64'd3000000 * 100);

        if (!finished_flag) begin
            $fatal(1, "[ERROR] Global timeout exceeded");
        end
    end

endmodule
