`timescale 1ns / 1ps

module tb_sliding_window;

    localparam CLK_PERIOD = 20; // 50MHz

    reg         clk;
    reg         rst_n;
    reg  [15:0] sensor_data;
    reg         data_valid;
    wire [15:0] window [0:3][0:3];
    wire        window_ready;

    sliding_window uut (
        .clk(clk),
        .rst_n(rst_n),
        .sensor_data(sensor_data),
        .data_valid(data_valid),
        .window(window),
        .window_ready(window_ready)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // 센서 데이터 1개 전송 태스크
    task send_data(input [15:0] val);
        begin
            @(posedge clk);
            sensor_data = val;
            data_valid  = 1'b1;
            @(posedge clk);
            data_valid  = 1'b0;
        end
    endtask

    // 1행(센서 4개) 전송 태스크
    task send_row(input [15:0] s0, s1, s2, s3);
        begin
            send_data(s0);
            send_data(s1);
            send_data(s2);
            send_data(s3);
        end
    endtask

    // 검증용 태스크: window[row][col]이 expected와 일치하는지 확인
    task check_cell(input integer row, col, input [15:0] expected, inout reg all_pass);
        begin
            if (window[row][col] == expected)
                $display("[PASS] window[%0d][%0d] = 0x%04X", row, col, window[row][col]);
            else begin
                $display("[FAIL] window[%0d][%0d] = 0x%04X (expected 0x%04X)",
                         row, col, window[row][col], expected);
                all_pass = 0;
            end
        end
    endtask

    integer i, j;
    reg all_pass;

    initial begin
        rst_n       = 0;
        sensor_data = 0;
        data_valid  = 0;
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        // =============================================================
        // Test 1: 최초 4행 채움 전 window_ready가 안 뜨는지 확인
        // =============================================================
        $display("=== Test 1: No ready before 4 rows ===");

        // 행 A: [0x0100, 0x0200, 0x0300, 0x0400]
        send_row(16'h0100, 16'h0200, 16'h0300, 16'h0400);
        @(posedge clk);
        if (window_ready)
            $display("[FAIL] window_ready asserted after only 1 row!");
        else
            $display("[PASS] window_ready not asserted after 1 row");

        // 행 B: [0x0500, 0x0600, 0x0700, 0x0800]
        send_row(16'h0500, 16'h0600, 16'h0700, 16'h0800);
        @(posedge clk);
        if (window_ready)
            $display("[FAIL] window_ready asserted after only 2 rows!");
        else
            $display("[PASS] window_ready not asserted after 2 rows");

        // 행 C: [0x0900, 0x0A00, 0x0B00, 0x0C00]
        send_row(16'h0900, 16'h0A00, 16'h0B00, 16'h0C00);
        @(posedge clk);
        if (window_ready)
            $display("[FAIL] window_ready asserted after only 3 rows!");
        else
            $display("[PASS] window_ready not asserted after 3 rows");

        $display(">>> Test 1 PASSED <<<");
        #(CLK_PERIOD * 3);

        // =============================================================
        // Test 2: 4번째 행 입력 → 최초 window_ready 발생 및 내용 검증
        //   기대 결과 (행0=가장 오래된, 행3=최신):
        //   행0: A = [0x0100, 0x0200, 0x0300, 0x0400]
        //   행1: B = [0x0500, 0x0600, 0x0700, 0x0800]
        //   행2: C = [0x0900, 0x0A00, 0x0B00, 0x0C00]
        //   행3: D = [0x0D00, 0x0E00, 0x0F00, 0x1000]
        // =============================================================
        $display("=== Test 2: First full window ===");

        // 행 D: [0x0D00, 0x0E00, 0x0F00, 0x1000]
        send_row(16'h0D00, 16'h0E00, 16'h0F00, 16'h1000);
        @(posedge clk);
        @(posedge clk);

        all_pass = 1;
        // 행0 = A (가장 오래된)
        check_cell(0, 0, 16'h0100, all_pass);
        check_cell(0, 1, 16'h0200, all_pass);
        check_cell(0, 2, 16'h0300, all_pass);
        check_cell(0, 3, 16'h0400, all_pass);
        // 행1 = B
        check_cell(1, 0, 16'h0500, all_pass);
        check_cell(1, 1, 16'h0600, all_pass);
        check_cell(1, 2, 16'h0700, all_pass);
        check_cell(1, 3, 16'h0800, all_pass);
        // 행2 = C
        check_cell(2, 0, 16'h0900, all_pass);
        check_cell(2, 1, 16'h0A00, all_pass);
        check_cell(2, 2, 16'h0B00, all_pass);
        check_cell(2, 3, 16'h0C00, all_pass);
        // 행3 = D (최신)
        check_cell(3, 0, 16'h0D00, all_pass);
        check_cell(3, 1, 16'h0E00, all_pass);
        check_cell(3, 2, 16'h0F00, all_pass);
        check_cell(3, 3, 16'h1000, all_pass);

        if (all_pass) $display(">>> Test 2 PASSED <<<");
        else          $display(">>> Test 2 FAILED <<<");
        #(CLK_PERIOD * 3);

        // =============================================================
        // Test 3: 5번째 행 입력 → FIFO 시프트 검증
        //   행 E = [0xAA00, 0xBB00, 0xCC00, 0xDD00] 입력
        //   기대 결과:
        //   행0: B = [0x0500, 0x0600, 0x0700, 0x0800]  ← A가 밀려남
        //   행1: C = [0x0900, 0x0A00, 0x0B00, 0x0C00]
        //   행2: D = [0x0D00, 0x0E00, 0x0F00, 0x1000]
        //   행3: E = [0xAA00, 0xBB00, 0xCC00, 0xDD00]  ← 새 행
        // =============================================================
        $display("=== Test 3: FIFO shift (5th row) ===");

        send_row(16'hAA00, 16'hBB00, 16'hCC00, 16'hDD00);
        @(posedge clk);
        @(posedge clk);

        all_pass = 1;
        // 행0 = B (A가 밀려서 사라짐)
        check_cell(0, 0, 16'h0500, all_pass);
        check_cell(0, 1, 16'h0600, all_pass);
        check_cell(0, 2, 16'h0700, all_pass);
        check_cell(0, 3, 16'h0800, all_pass);
        // 행1 = C
        check_cell(1, 0, 16'h0900, all_pass);
        check_cell(1, 1, 16'h0A00, all_pass);
        check_cell(1, 2, 16'h0B00, all_pass);
        check_cell(1, 3, 16'h0C00, all_pass);
        // 행2 = D
        check_cell(2, 0, 16'h0D00, all_pass);
        check_cell(2, 1, 16'h0E00, all_pass);
        check_cell(2, 2, 16'h0F00, all_pass);
        check_cell(2, 3, 16'h1000, all_pass);
        // 행3 = E (최신)
        check_cell(3, 0, 16'hAA00, all_pass);
        check_cell(3, 1, 16'hBB00, all_pass);
        check_cell(3, 2, 16'hCC00, all_pass);
        check_cell(3, 3, 16'hDD00, all_pass);

        if (all_pass) $display(">>> Test 3 PASSED <<<");
        else          $display(">>> Test 3 FAILED <<<");
        #(CLK_PERIOD * 3);

        // =============================================================
        // Test 4: 6번째 행 입력 → 연속 FIFO 시프트 검증
        //   행 F = [0x1100, 0x2200, 0x3300, 0x4400] 입력
        //   기대 결과:
        //   행0: C = [0x0900, 0x0A00, 0x0B00, 0x0C00]  ← B가 밀려남
        //   행1: D = [0x0D00, 0x0E00, 0x0F00, 0x1000]
        //   행2: E = [0xAA00, 0xBB00, 0xCC00, 0xDD00]
        //   행3: F = [0x1100, 0x2200, 0x3300, 0x4400]  ← 새 행
        // =============================================================
        $display("=== Test 4: FIFO shift (6th row) ===");

        send_row(16'h1100, 16'h2200, 16'h3300, 16'h4400);
        @(posedge clk);
        @(posedge clk);

        all_pass = 1;
        // 행0 = C
        check_cell(0, 0, 16'h0900, all_pass);
        check_cell(0, 1, 16'h0A00, all_pass);
        check_cell(0, 2, 16'h0B00, all_pass);
        check_cell(0, 3, 16'h0C00, all_pass);
        // 행1 = D
        check_cell(1, 0, 16'h0D00, all_pass);
        check_cell(1, 1, 16'h0E00, all_pass);
        check_cell(1, 2, 16'h0F00, all_pass);
        check_cell(1, 3, 16'h1000, all_pass);
        // 행2 = E
        check_cell(2, 0, 16'hAA00, all_pass);
        check_cell(2, 1, 16'hBB00, all_pass);
        check_cell(2, 2, 16'hCC00, all_pass);
        check_cell(2, 3, 16'hDD00, all_pass);
        // 행3 = F (최신)
        check_cell(3, 0, 16'h1100, all_pass);
        check_cell(3, 1, 16'h2200, all_pass);
        check_cell(3, 2, 16'h3300, all_pass);
        check_cell(3, 3, 16'h4400, all_pass);

        if (all_pass) $display(">>> Test 4 PASSED <<<");
        else          $display(">>> Test 4 FAILED <<<");
        #(CLK_PERIOD * 3);

        // =============================================================
        // Test 5: 리셋 후 다시 4행 채워야 ready가 뜨는지 확인
        // =============================================================
        $display("=== Test 5: Reset and refill ===");

        rst_n = 0;
        #(CLK_PERIOD * 3);
        rst_n = 1;
        #(CLK_PERIOD * 3);

        // 3행만 보냄 → ready 안 떠야 함
        send_row(16'h0100, 16'h0100, 16'h0100, 16'h0100);
        send_row(16'h0200, 16'h0200, 16'h0200, 16'h0200);
        send_row(16'h0300, 16'h0300, 16'h0300, 16'h0300);
        @(posedge clk);
        @(posedge clk);

        if (window_ready)
            $display("[FAIL] window_ready after reset with only 3 rows!");
        else
            $display("[PASS] No window_ready after reset with 3 rows");

        // 4번째 행 → ready 떠야 함
        send_row(16'h0400, 16'h0400, 16'h0400, 16'h0400);
        @(posedge clk);
        @(posedge clk);

        all_pass = 1;
        check_cell(0, 0, 16'h0100, all_pass);
        check_cell(1, 0, 16'h0200, all_pass);
        check_cell(2, 0, 16'h0300, all_pass);
        check_cell(3, 0, 16'h0400, all_pass);

        if (all_pass) $display(">>> Test 5 PASSED <<<");
        else          $display(">>> Test 5 FAILED <<<");

        #(CLK_PERIOD * 5);
        $display("=== All tests done ===");
        $stop;
    end

    // window_ready 모니터
    always @(posedge window_ready) begin
        $display("[%0t ns] window_ready asserted!", $time);
    end

endmodule
