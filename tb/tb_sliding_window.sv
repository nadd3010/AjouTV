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

    // 데이터 1개 전송 태스크
    task send_data(input [15:0] val);
        begin
            @(posedge clk);
            sensor_data = val;
            data_valid  = 1'b1;
            @(posedge clk);
            data_valid  = 1'b0;
        end
    endtask

    integer i, j;
    reg [15:0] expected;
    reg all_pass;

    initial begin
        rst_n       = 0;
        sensor_data = 0;
        data_valid  = 0;
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        // ===== Test 1: 순차 데이터 0x0100 ~ 0x1000 =====
        $display("=== Test 1: Sequential data ===");
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                send_data((i * 4 + j + 1) * 16'h0100);
                // 예: [0][0]=0x0100, [0][1]=0x0200, ... [3][3]=0x1000
            end
        end

        @(posedge clk); // window_ready 확인 대기
        @(posedge clk);

        // 결과 검증
        all_pass = 1;
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                expected = (i * 4 + j + 1) * 16'h0100;
                if (window[i][j] == expected)
                    $display("[PASS] window[%0d][%0d] = 0x%04X", i, j, window[i][j]);
                else begin
                    $display("[FAIL] window[%0d][%0d] = 0x%04X (expected 0x%04X)", i, j, window[i][j], expected);
                    all_pass = 0;
                end
            end
        end

        if (all_pass)
            $display(">>> Test 1 PASSED <<<");
        else
            $display(">>> Test 1 FAILED <<<");

        #(CLK_PERIOD * 5);

        // ===== Test 2: 두 번째 윈도우 (연속 전송) =====
        $display("=== Test 2: Second window ===");
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                send_data(16'hFF00);  // 모든 값 255.0 (Q8.8)
            end
        end

        @(posedge clk);
        @(posedge clk);

        all_pass = 1;
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                if (window[i][j] == 16'hFF00)
                    $display("[PASS] window[%0d][%0d] = 0x%04X", i, j, window[i][j]);
                else begin
                    $display("[FAIL] window[%0d][%0d] = 0x%04X (expected 0xFF00)", i, j, window[i][j]);
                    all_pass = 0;
                end
            end
        end

        if (all_pass)
            $display(">>> Test 2 PASSED <<<");
        else
            $display(">>> Test 2 FAILED <<<");

        #(CLK_PERIOD * 5);
        $display("=== All tests done ===");
        $stop;
    end

    // window_ready 모니터
    always @(posedge window_ready) begin
        $display("[%0t ns] window_ready asserted!", $time);
    end

endmodule
