`timescale 1ns / 1ps

module tb_uart_tx;

    localparam CLK_PERIOD = 20;
    localparam BIT_PERIOD = 8680;

    reg         clk;
    reg         rst_n;
    reg         start;
    reg         label;
    reg  [17:0] col_sum [0:3];
    wire        tx;
    wire        done;

    uart_tx uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .label(label),
        .col_sum(col_sum),
        .tx(tx),
        .done(done)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // TX 라인에서 1바이트 캡처
    task capture_byte(output [7:0] captured);
        integer i;
        begin
            @(negedge tx);           // Start bit 감지
            #(BIT_PERIOD / 2);       // Start bit 중앙
            for (i = 0; i < 8; i = i + 1) begin
                #(BIT_PERIOD);
                captured[i] = tx;
            end
            #(BIT_PERIOD);           // Stop bit
        end
    endtask

    reg [7:0] cap [0:8];            // 캡처된 9바이트
    reg [7:0] exp [0:8];            // 기대값 9바이트
    integer k;
    reg all_pass;

    // 9바이트 캡처 + 검증 태스크
    task test_packet(
        input        t_label,
        input [17:0] t_cs0,
        input [17:0] t_cs1,
        input [17:0] t_cs2,
        input [17:0] t_cs3
    );
        begin
            // 입력 설정
            @(posedge clk);
            label      = t_label;
            col_sum[0] = t_cs0;
            col_sum[1] = t_cs1;
            col_sum[2] = t_cs2;
            col_sum[3] = t_cs3;
            start      = 1'b1;
            @(posedge clk);
            start      = 1'b0;

            // 기대값 구성 (상위 2비트 버림 → [15:0] 전송)
            exp[0] = {7'd0, t_label};
            exp[1] = t_cs0[15:8];
            exp[2] = t_cs0[7:0];
            exp[3] = t_cs1[15:8];
            exp[4] = t_cs1[7:0];
            exp[5] = t_cs2[15:8];
            exp[6] = t_cs2[7:0];
            exp[7] = t_cs3[15:8];
            exp[8] = t_cs3[7:0];

            // 9바이트 캡처
            for (k = 0; k < 9; k = k + 1) begin
                capture_byte(cap[k]);
            end

            // done 신호 대기
            @(posedge done);
            @(posedge clk);

            // 결과 검증
            all_pass = 1;
            for (k = 0; k < 9; k = k + 1) begin
                if (cap[k] == exp[k])
                    $display("[PASS] byte[%0d] = 0x%02X (expected 0x%02X)", k, cap[k], exp[k]);
                else begin
                    $display("[FAIL] byte[%0d] = 0x%02X (expected 0x%02X)", k, cap[k], exp[k]);
                    all_pass = 0;
                end
            end

            if (all_pass)
                $display(">>> PACKET PASSED <<<");
            else
                $display(">>> PACKET FAILED <<<");
        end
    endtask

    initial begin
        rst_n  = 0;
        start  = 0;
        label  = 0;
        col_sum[0] = 18'd0;
        col_sum[1] = 18'd0;
        col_sum[2] = 18'd0;
        col_sum[3] = 18'd0;
        #(CLK_PERIOD * 10);
        rst_n = 1;
        #(CLK_PERIOD * 10);

        // ===== Test 1: 정상 판정, 다양한 col_sum =====
        $display("=== Test 1: label=0, col_sum = 0x0100, 0x0200, 0x0300, 0x0400 ===");
        test_packet(
            1'b0,
            18'h00100,   // col_sum[0] = 0x0100 (Q8.8: 1.0)
            18'h00200,   // col_sum[1] = 0x0200 (Q8.8: 2.0)
            18'h00300,   // col_sum[2] = 0x0300 (Q8.8: 3.0)
            18'h00400    // col_sum[3] = 0x0400 (Q8.8: 4.0)
        );
        #(CLK_PERIOD * 20);

        // ===== Test 2: 이상 판정, 모든 col_sum 동일 =====
        $display("=== Test 2: label=1, col_sum all 0xFF00 ===");
        test_packet(
            1'b1,
            18'h0FF00,
            18'h0FF00,
            18'h0FF00,
            18'h0FF00
        );
        #(CLK_PERIOD * 20);

        // ===== Test 3: 경계값 0x0000 =====
        $display("=== Test 3: label=0, col_sum all 0x0000 ===");
        test_packet(
            1'b0,
            18'h00000,
            18'h00000,
            18'h00000,
            18'h00000
        );
        #(CLK_PERIOD * 20);

        // ===== Test 4: 18비트 상위 비트 버림 확인 =====
        $display("=== Test 4: 18-bit overflow test (0x3ABCD -> transmit 0xABCD) ===");
        test_packet(
            1'b1,
            18'h3ABCD,   // 상위 2비트(0x3) 버림 → 0xABCD 전송
            18'h21234,   // 상위 2비트(0x2) 버림 → 0x1234 전송
            18'h15678,   // 상위 2비트(0x1) 버림 → 0x5678 전송
            18'h09ABC    // 상위 2비트(0x0) 버림 → 0x9ABC 전송
        );
        #(CLK_PERIOD * 20);

        $display("=== All tests done ===");
        $stop;
    end

endmodule
