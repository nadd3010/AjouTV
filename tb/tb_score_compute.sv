
`timescale 1ns / 1ps

module tb_score_compute;

    // =====================================================
    // 1. 입출력 신호 선언
    // =====================================================
    logic clk;
    logic rst_n;
    logic start;

    logic [15:0] q [0:3][0:3];
    logic [15:0] k [0:3][0:3];

    // RTL의 signed 출력 포트와 타입 일치
    logic signed [15:0] score [0:3][0:3];
    logic done;

    // 예상 결과
    logic signed [15:0] expected [0:3][0:3];

    integer cycles;
    integer errors;

    // =====================================================
    // 2. DUT 연결
    // =====================================================
    score_compute dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .q(q),
        .k(k),
        .score(score),
        .done(done)
    );

    // 100MHz Clock
    always #5 clk = ~clk;

    // =====================================================
    // 3. 테스트 시나리오
    // =====================================================
    initial begin

        clk = 0;
        rst_n = 0;
        start = 0;

        cycles = 0;
        errors = 0;

        // 입력 및 예상 결과 초기화
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                q[i][j] = 16'h0000;
                k[i][j] = 16'h0000;
                expected[i][j] = 16'h0000;
            end
        end

        // -------------------------------------------------
        // Reset 해제
        // -------------------------------------------------
        repeat (3) @(negedge clk);
        rst_n = 1;

        // -------------------------------------------------
        // Test 1: 양수 연산
        //
        // Q[0] = [1.0, 2.0, 0, 0]
        // K[0] = [1.0, 0.5, 0, 0]
        //
        // Dot Product = 1.0 + 1.0 = 2.0
        // Scaling = 2.0 / 2 = 1.0
        // Expected = 16'h0100
        // -------------------------------------------------
        q[0][0] = 16'h0100;
        q[0][1] = 16'h0200;

        k[0][0] = 16'h0100;
        k[0][1] = 16'h0080;

        expected[0][0] = 16'h0100;

        // -------------------------------------------------
        // Test 2: 음수 연산
        //
        // Q[1] = [-1.0, 0, 0, 0]
        // K[0] = [ 1.0, 0.5, 0, 0]
        //
        // Dot Product = -1.0
        // Scaling = -0.5
        // Expected = 16'hFF80
        // -------------------------------------------------
        q[1][0] = 16'hFF00;

        expected[1][0] = 16'hFF80;

        // -------------------------------------------------
        // Test 3: 다른 행/열 연산
        //
        // Q[2][2] =  1.0
        // K[2][2] = -2.0
        //
        // Dot Product = -2.0
        // Scaling = -1.0
        // Expected = 16'hFF00
        // -------------------------------------------------
        q[2][2] = 16'h0100;
        k[2][2] = 16'hFE00;

        expected[2][2] = 16'hFF00;

        // -------------------------------------------------
        // Start: 1클럭 펄스
        // -------------------------------------------------
        @(negedge clk);
        start = 1;

        @(negedge clk);
        start = 0;

        $display("========================================");
        $display(" Score Compute Pipeline Test");
        $display(" Clock : 100MHz (10ns)");
        $display("========================================");

        // -------------------------------------------------
        // done 대기 (최대 100클럭)
        // -------------------------------------------------
        while (done !== 1'b1 && cycles < 100) begin
            @(negedge clk);
            cycles = cycles + 1;
        end

        if (done !== 1'b1) begin
            $fatal(1, "[FAIL] Timeout! done not asserted.");
        end

        $display("[INFO] done asserted.");
        $display("[INFO] Wait cycles = %0d", cycles);

        // -------------------------------------------------
        // 4x4 결과 자동 비교
        // -------------------------------------------------
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin

                if (score[i][j] !== expected[i][j]) begin

                    $display(
                        "[FAIL] score[%0d][%0d] = %h, expected = %h",
                        i, j, score[i][j], expected[i][j]
                    );

                    errors = errors + 1;

                end else begin

                    $display(
                        "[PASS] score[%0d][%0d] = %h",
                        i, j, score[i][j]
                    );

                end
            end
        end

        // -------------------------------------------------
        // done이 1클럭 펄스인지 확인
        // -------------------------------------------------
        @(negedge clk);

        if (done !== 1'b0) begin
            $display("[FAIL] done pulse width error!");
            errors = errors + 1;
        end

        // -------------------------------------------------
        // 최종 결과
        // -------------------------------------------------
        $display("========================================");
        $display(" Total output elements : 16");
        $display(" Mismatch count        : %0d", errors);

        if (errors == 0) begin
            $display(" [RESULT] ALL PASS");
        end else begin
            $display(" [RESULT] FAIL");
        end

        $display("========================================");

        if (errors != 0)
            $fatal(1, "Score Compute Test FAILED");

        $finish;
    end

endmodule
