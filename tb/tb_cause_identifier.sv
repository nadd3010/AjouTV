
`timescale 1ns / 1ps

module tb_cause_identifier;

    // =====================================================
    // 1. 입출력 신호
    // =====================================================
    logic clk;
    logic rst_n;

    // RTL 포트와 signed 속성 일치
    logic signed [15:0] attn_weight [0:3][0:3];

    logic [17:0] col_sum [0:3];
    logic [1:0]  cause_index;

    integer errors;

    // =====================================================
    // 2. DUT
    // =====================================================
    cause_identifier dut (
        .clk(clk),
        .rst_n(rst_n),
        .attn_weight(attn_weight),
        .col_sum(col_sum),
        .cause_index(cause_index)
    );

    // 100MHz Clock
    always #5 clk = ~clk;

    // =====================================================
    // 3. 자동 검증 Task
    //
    // Clock 1: col_sum 갱신
    // Clock 2: cause_index 갱신
    // =====================================================
    task automatic check_case (
        input string test_name,
        input logic [17:0] exp0,
        input logic [17:0] exp1,
        input logic [17:0] exp2,
        input logic [17:0] exp3,
        input logic [1:0]  exp_idx
    );

        integer mism;

        begin
            mism = 0;

            // 2단계 파이프라인 완료 대기
            repeat (2) @(posedge clk);
            #1;

            $display("========================================");
            $display("%s", test_name);

            if (col_sum[0] !== exp0) begin
                $display("[FAIL] col_sum[0] = %0d, expected = %0d",
                         col_sum[0], exp0);
                mism++;
            end

            if (col_sum[1] !== exp1) begin
                $display("[FAIL] col_sum[1] = %0d, expected = %0d",
                         col_sum[1], exp1);
                mism++;
            end

            if (col_sum[2] !== exp2) begin
                $display("[FAIL] col_sum[2] = %0d, expected = %0d",
                         col_sum[2], exp2);
                mism++;
            end

            if (col_sum[3] !== exp3) begin
                $display("[FAIL] col_sum[3] = %0d, expected = %0d",
                         col_sum[3], exp3);
                mism++;
            end

            if (cause_index !== exp_idx) begin
                $display("[FAIL] cause_index = %0d, expected = %0d",
                         cause_index, exp_idx);
                mism++;
            end

            $display("Column Sum = [%0d, %0d, %0d, %0d]",
                     col_sum[0], col_sum[1],
                     col_sum[2], col_sum[3]);

            $display("Cause Index = %0d", cause_index);

            if (mism == 0)
                $display("[PASS] %s", test_name);
            else
                $display("[FAIL] %s (%0d mismatches)",
                         test_name, mism);

            errors = errors + mism;

            $display("========================================");
        end

    endtask

    // =====================================================
    // 4. Test Scenarios
    // =====================================================
    initial begin

        clk    = 0;
        rst_n  = 0;
        errors = 0;

        // 입력 초기화
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                attn_weight[i][j] = 16'd0;
            end
        end

        // Reset
        repeat (3) @(negedge clk);
        rst_n = 1;

        // -------------------------------------------------
        // Test Case 1: Sensor 1 최대
        //
        // Expected:
        // col_sum = [40, 4000, 80, 20]
        // cause_index = 1
        // -------------------------------------------------
        @(negedge clk);

        for (int i = 0; i < 4; i++) begin
            attn_weight[i][0] = 16'd10;
            attn_weight[i][1] = 16'd1000;
            attn_weight[i][2] = 16'd20;
            attn_weight[i][3] = 16'd5;
        end

        check_case(
            "Test 1: Sensor 1 Max",
            18'd40,
            18'd4000,
            18'd80,
            18'd20,
            2'd1
        );

        // -------------------------------------------------
        // Test Case 2: Overflow Check
        //
        // unsigned 합산 기준
        // col_sum[0] = 262139
        // col_sum[3] = 262140
        //
        // Expected cause_index = 3
        // -------------------------------------------------
        @(negedge clk);

        for (int i = 0; i < 4; i++) begin
            attn_weight[i][0] = 16'hFFFF;
            attn_weight[i][1] = 16'h1000;
            attn_weight[i][2] = 16'h2000;
            attn_weight[i][3] = 16'hFFFF;
        end

        attn_weight[3][0] = 16'hFFFE;

        check_case(
            "Test 2: Overflow & Sensor 3 Max",
            18'd262139,
            18'd16384,
            18'd32768,
            18'd262140,
            2'd3
        );

        // -------------------------------------------------
        // Test Case 3: Sensor 2 최대
        //
        // Expected:
        // col_sum = [1200, 3200, 8000, 600]
        // cause_index = 2
        // -------------------------------------------------
        @(negedge clk);

        for (int i = 0; i < 4; i++) begin
            attn_weight[i][0] = 16'd300;
            attn_weight[i][1] = 16'd800;
            attn_weight[i][2] = 16'd2000;
            attn_weight[i][3] = 16'd150;
        end

        check_case(
            "Test 3: Sensor 2 Max",
            18'd1200,
            18'd3200,
            18'd8000,
            18'd600,
            2'd2
        );

        // -------------------------------------------------
        // Test Case 4: 동일한 최대값
        //
        // Sensor 0과 Sensor 2가 동점이면
        // 작은 Index인 0을 선택해야 함
        //
        // Expected:
        // col_sum = [400, 200, 400, 100]
        // cause_index = 0
        // -------------------------------------------------
        @(negedge clk);

        for (int i = 0; i < 4; i++) begin
            attn_weight[i][0] = 16'd100;
            attn_weight[i][1] = 16'd50;
            attn_weight[i][2] = 16'd100;
            attn_weight[i][3] = 16'd25;
        end

        check_case(
            "Test 4: Tie-Break (Lowest Index)",
            18'd400,
            18'd200,
            18'd400,
            18'd100,
            2'd0
        );

        // =================================================
        // 5. Final Result
        // =================================================
        $display("");
        $display("========================================");
        $display(" Cause Identifier Pipeline Test");
        $display(" Clock : 100MHz");
        $display(" Total Test Cases : 4");
        $display(" Total Mismatches : %0d", errors);

        if (errors == 0)
            $display(" [RESULT] ALL PASS");
        else
            $display(" [RESULT] FAIL");

        $display("========================================");

        if (errors != 0)
            $fatal(1, "Cause Identifier Test FAILED");

        $finish;
    end

endmodule
