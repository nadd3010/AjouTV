
`timescale 1ns / 1ps

module tb_qkv_transform;

    // =====================================================
    // 1. 입출력 신호
    // =====================================================
    logic clk;
    logic rst_n;
    logic start;

    logic [15:0] x [0:3][0:3];

    logic [15:0] q [0:3][0:3];
    logic [15:0] k [0:3][0:3];
    logic [15:0] v [0:3][0:3];

    logic done;

    // 예상 결과
    logic [15:0] expected_q [0:3][0:3];
    logic [15:0] expected_k [0:3][0:3];
    logic [15:0] expected_v [0:3][0:3];

    logic signed [33:0] acc_q;
    logic signed [33:0] acc_k;
    logic signed [33:0] acc_v;

    integer cycles;
    integer errors;

    // =====================================================
    // 2. DUT 연결
    // =====================================================
    qkv_transform dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .x(x),
        .q(q),
        .k(k),
        .v(v),
        .done(done)
    );

    // 100MHz clock
    always #5 clk = ~clk;

    // =====================================================
    // 3. 테스트
    // =====================================================
    initial begin

        clk = 0;
        rst_n = 0;
        start = 0;

        cycles = 0;
        errors = 0;

        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                x[i][j] = 16'h0000;
                expected_q[i][j] = 16'h0000;
                expected_k[i][j] = 16'h0000;
                expected_v[i][j] = 16'h0000;
            end
        end

        // -------------------------------------------------
        // Reset
        // -------------------------------------------------
        repeat (3) @(negedge clk);
        rst_n = 1;

        // 기존 테스트 데이터 유지
        x[0][0] = 16'h0100; // 1.0
        x[1][1] = 16'h0280; // 2.5

        // -------------------------------------------------
        // ROM 데이터 확인
        // -------------------------------------------------
        $display("========================================");
        $display(" QKV Transform Pipeline Test");
        $display(" Clock : 100MHz (10ns)");
        $display("========================================");

        $display("=== Loaded weights_rom ===");

        for (int i = 0; i < 48; i++) begin
            $display("weights_rom[%0d] = %h",
                     i, dut.weights_rom[i]);

            if (^dut.weights_rom[i] === 1'bx) begin
                $fatal(1,
                    "[FAIL] ROM contains X/Z at index %0d", i);
            end
        end

        // -------------------------------------------------
        // Golden Reference 계산
        //
        // Q = X * W_Q
        // K = X * W_K
        // V = X * W_V
        //
        // 각 연산은 signed 16x16 곱셈을 누적한 뒤
        // [23:8] 비트를 출력
        // -------------------------------------------------
        for (int r = 0; r < 4; r++) begin
            for (int c = 0; c < 4; c++) begin

                acc_q = 34'sd0;
                acc_k = 34'sd0;
                acc_v = 34'sd0;

                for (int d = 0; d < 4; d++) begin

                    acc_q = acc_q +
                        ($signed(x[r][d]) *
                         $signed(dut.weights_rom[d*4+c]));

                    acc_k = acc_k +
                        ($signed(x[r][d]) *
                         $signed(dut.weights_rom[16+d*4+c]));

                    acc_v = acc_v +
                        ($signed(x[r][d]) *
                         $signed(dut.weights_rom[32+d*4+c]));

                end

                expected_q[r][c] = acc_q[23:8];
                expected_k[r][c] = acc_k[23:8];
                expected_v[r][c] = acc_v[23:8];

            end
        end

        // -------------------------------------------------
        // Start pulse
        // -------------------------------------------------
        @(negedge clk);
        start = 1'b1;

        @(negedge clk);
        start = 1'b0;

        // -------------------------------------------------
        // Wait for done (timeout: 100 clocks)
        // -------------------------------------------------
        while (done !== 1'b1 && cycles < 100) begin
            @(negedge clk);
            cycles = cycles + 1;
        end

        if (done !== 1'b1)
            $fatal(1, "[FAIL] Timeout: done not asserted");

        $display("[INFO] done asserted");
        $display("[INFO] Wait cycles = %0d", cycles);

        // -------------------------------------------------
        // Compare all 48 outputs
        // -------------------------------------------------
        for (int r = 0; r < 4; r++) begin
            for (int c = 0; c < 4; c++) begin

                if (q[r][c] !== expected_q[r][c]) begin
                    $display(
                        "[FAIL] Q[%0d][%0d] = %h, expected = %h",
                        r, c, q[r][c], expected_q[r][c]);
                    errors++;
                end

                if (k[r][c] !== expected_k[r][c]) begin
                    $display(
                        "[FAIL] K[%0d][%0d] = %h, expected = %h",
                        r, c, k[r][c], expected_k[r][c]);
                    errors++;
                end

                if (v[r][c] !== expected_v[r][c]) begin
                    $display(
                        "[FAIL] V[%0d][%0d] = %h, expected = %h",
                        r, c, v[r][c], expected_v[r][c]);
                    errors++;
                end

            end
        end

        // -------------------------------------------------
        // done pulse width 확인
        // -------------------------------------------------
        @(negedge clk);

        if (done !== 1'b0) begin
            $display("[FAIL] done pulse width error");
            errors++;
        end

        // -------------------------------------------------
        // Result
        // -------------------------------------------------
        $display("========================================");
        $display(" Total output elements : 48");
        $display(" Mismatch count        : %0d", errors);

        if (errors == 0)
            $display(" [RESULT] ALL PASS");
        else
            $display(" [RESULT] FAIL");

        $display("========================================");

        if (errors != 0)
            $fatal(1, "QKV Transform Test FAILED");

        $finish;
    end

endmodule
