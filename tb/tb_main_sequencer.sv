`timescale 1ns / 1ps

module tb_main_sequencer;

    localparam CLK_PERIOD = 20;

    reg        clk;
    reg        rst_n;
    reg        window_ready;
    reg        attn_done;
    reg        classify_done;
    reg        tx_done;
    wire       attn_start;
    wire       classify_start;
    wire       tx_start;
    wire [2:0] state;

    main_sequencer uut (
        .clk(clk),
        .rst_n(rst_n),
        .window_ready(window_ready),
        .attn_done(attn_done),
        .classify_done(classify_done),
        .tx_done(tx_done),
        .attn_start(attn_start),
        .classify_start(classify_start),
        .tx_start(tx_start),
        .state(state)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // 1클럭 펄스 생성 태스크
    task pulse(input string name);
        begin
            @(posedge clk);
            case (name)
                "window_ready":  window_ready  = 1'b1;
                "attn_done":     attn_done     = 1'b1;
                "classify_done": classify_done = 1'b1;
                "tx_done":       tx_done       = 1'b1;
            endcase
            @(posedge clk);
            window_ready  = 1'b0;
            attn_done     = 1'b0;
            classify_done = 1'b0;
            tx_done       = 1'b0;
        end
    endtask

    integer pass_cnt;
    integer fail_cnt;

    task check_state(input [2:0] expected, input string label);
        begin
            @(posedge clk);
            if (state === expected) begin
                $display("[PASS] %s: state = %0d (expected %0d)", label, state, expected);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] %s: state = %0d (expected %0d)", label, state, expected);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    task check_signal(input logic actual, input logic expected, input string name);
        begin
            if (actual === expected) begin
                $display("[PASS] %s = %0b (expected %0b)", name, actual, expected);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] %s = %0b (expected %0b)", name, actual, expected);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    initial begin
        pass_cnt      = 0;
        fail_cnt      = 0;
        rst_n         = 0;
        window_ready  = 0;
        attn_done     = 0;
        classify_done = 0;
        tx_done       = 0;
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        // ===== Test 1: 리셋 후 IDLE =====
        $display("=== Test 1: After reset ===");
        check_state(3'd0, "IDLE");

        // ===== Test 2: window_ready → ATTENTION =====
        $display("=== Test 2: window_ready -> ATTENTION ===");
        pulse("window_ready");
        check_state(3'd1, "ATTENTION");

        // attn_start 펄스 확인
        // (이미 1클럭 지남 → 펄스 끝났을 수 있으므로 상태로 검증)

        // ===== Test 3: ATTENTION에서 다른 신호 무시 =====
        $display("=== Test 3: Ignore other signals in ATTENTION ===");
        pulse("classify_done");
        check_state(3'd1, "Still ATTENTION");
        pulse("tx_done");
        check_state(3'd1, "Still ATTENTION");

        // ===== Test 4: attn_done → CLASSIFY =====
        $display("=== Test 4: attn_done -> CLASSIFY ===");
        pulse("attn_done");
        check_state(3'd2, "CLASSIFY");

        // ===== Test 5: classify_done → OUTPUT =====
        $display("=== Test 5: classify_done -> OUTPUT ===");
        pulse("classify_done");
        check_state(3'd3, "OUTPUT");

        // ===== Test 6: tx_done → IDLE 복귀 =====
        $display("=== Test 6: tx_done -> IDLE ===");
        pulse("tx_done");
        check_state(3'd0, "Back to IDLE");

        // ===== Test 7: 전체 사이클 2회 연속 =====
        $display("=== Test 7: Full cycle repeat ===");
        pulse("window_ready");
        check_state(3'd1, "2nd ATTENTION");
        pulse("attn_done");
        check_state(3'd2, "2nd CLASSIFY");
        pulse("classify_done");
        check_state(3'd3, "2nd OUTPUT");
        pulse("tx_done");
        check_state(3'd0, "2nd IDLE");

        // ===== Test 8: IDLE에서 잘못된 신호 무시 =====
        $display("=== Test 8: Ignore wrong signals in IDLE ===");
        pulse("attn_done");
        check_state(3'd0, "Still IDLE after attn_done");
        pulse("classify_done");
        check_state(3'd0, "Still IDLE after classify_done");
        pulse("tx_done");
        check_state(3'd0, "Still IDLE after tx_done");

        // ===== 결과 요약 =====
        $display("=================================");
        $display("Total: %0d PASS, %0d FAIL", pass_cnt, fail_cnt);
        $display("=================================");
        $stop;
    end

endmodule
