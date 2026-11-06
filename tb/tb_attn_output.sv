
`timescale 1ns / 1ps

module tb_attn_output;

    // ==================================================
    // 1. DUT 입출력 신호
    // ==================================================
    logic clk;
    logic rst_n;
    logic start;

    logic signed [15:0] attn_weight [0:3][0:3];
    logic [15:0] v [0:3][0:3];

    wire [15:0] attn_out [0:3][0:3];
    wire done;

    // Golden Model 기대값
    logic [15:0] expected [0:3][0:3];

    integer cycles;
    integer mismatch;

    // ==================================================
    // 2. DUT 연결
    // ==================================================
    attn_output dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .attn_weight(attn_weight),
        .v(v),
        .attn_out(attn_out),
        .done(done)
    );

    // ==================================================
    // 3. 100MHz 클럭
    // ==================================================
    initial clk = 0;
    always #5 clk = ~clk;

    // ==================================================
    // 4. 테스트 시나리오
    // ==================================================
    initial begin

        rst_n = 0;
        start = 0;
        cycles = 0;
        mismatch = 0;

        // 입력 및 기대값 초기화
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                attn_weight[i][j] = 16'h0000;
                v[i][j] = 16'h0000;
                expected[i][j] = 16'h0000;
            end
        end

        // Reset 해제
        repeat (3) @(negedge clk);
        rst_n = 1;

        // ==================================================
        // 5. 테스트 데이터 설정 (Q8.8)
        // ==================================================
        @(negedge clk);

        // Attention Weight
        attn_weight[0][0] = 16'h0100; // 1.0

        // V 행렬
        v[0][0] = 16'h0200; // 2.0
        v[0][1] = 16'h0300; // 3.0

        // 기대 출력값
        expected[0][0] = 16'h0200;
        expected[0][1] = 16'h0300;

        // 나머지 14개 원소는 0

        $display("=====================================");
        $display(" Attention Output Pipeline Test");
        $display(" Clock : 100MHz");
        $display("=====================================");

        // ==================================================
        // 6. Start 신호
        // ==================================================
        start = 1'b1;

        @(negedge clk);
        start = 1'b0;

        // ==================================================
        // 7. Done 신호 대기 (Timeout 100클럭)
        // ==================================================
        cycles = 0;

        while (done !== 1'b1 && cycles < 100) begin
            @(negedge clk);
            cycles = cycles + 1;
        end

        if (done !== 1'b1) begin
            $display("[FAIL] Timeout! done not asserted.");
            $fatal(1, "Attention Output Timeout");
        end

        $display("[INFO] done asserted.");
        $display("[INFO] Wait cycles = %0d", cycles);

        // ==================================================
        // 8. 출력값 자동 비교
        // ==================================================
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin

                if (attn_out[i][j] !== expected[i][j]) begin

                    mismatch = mismatch + 1;

                    $display(
                        "[MISMATCH] [%0d][%0d] RTL=%h EXP=%h",
                        i, j,
                        attn_out[i][j],
                        expected[i][j]
                    );

                end
            end
        end

        // ==================================================
        // 9. Done 펄스 폭 확인
        // ==================================================
        @(negedge clk);

        if (done !== 1'b0) begin
            mismatch = mismatch + 1;
            $display("[MISMATCH] done pulse exceeds 1 clock");
        end

        // ==================================================
        // 10. 최종 결과
        // ==================================================
        $display("=====================================");
        $display(" Total output elements : 16");
        $display(" Mismatch count        : %0d", mismatch);

        if (mismatch == 0) begin
            $display(" [RESULT] ALL PASS");
        end else begin
            $display(" [RESULT] FAIL");
            $fatal(1, "Attention Output Test Failed");
        end

        $display("=====================================");

        $stop;

    end

endmodule
