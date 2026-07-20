`timescale 1ns / 1ps

module tb_uart_rx;

    // 50MHz → 주기 20ns
    localparam CLK_PERIOD   = 20;
    // 115200 baud → 1비트 = 8680ns
    localparam BIT_PERIOD   = 8680;

    reg        clk;
    reg        rst_n;
    reg        rx;
    wire [7:0] data_out;
    wire       data_valid;

    // DUT 인스턴스
    uart_rx uut (
        .clk        (clk),
        .rst_n      (rst_n),
        .rx         (rx),
        .data_out   (data_out),
        .data_valid (data_valid)
    );

    // 클럭 생성
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // UART 1바이트 전송 태스크 (LSB first)
    task send_byte;
        input [7:0] data;
        integer i;
        begin
            // Start Bit
            rx = 1'b0;
            #(BIT_PERIOD);

            // Data Bits (D0 ~ D7)
            for (i = 0; i < 8; i = i + 1) begin
                rx = data[i];
                #(BIT_PERIOD);
            end

            // Stop Bit
            rx = 1'b1;
            #(BIT_PERIOD);
        end
    endtask

    // 테스트 시나리오
    initial begin
        // 초기화
        rst_n = 0;
        rx    = 1'b1;    // IDLE 상태는 HIGH
        #(CLK_PERIOD * 10);
        rst_n = 1;
        #(CLK_PERIOD * 10);

        // 테스트 1: 0x55 (01010101) 전송
        $display("=== Test 1: Sending 0x55 ===");
        send_byte(8'h55);
        #(BIT_PERIOD * 2);

        // 테스트 2: 0xA3 (10100011) 전송
        $display("=== Test 2: Sending 0xA3 ===");
        send_byte(8'hA3);
        #(BIT_PERIOD * 2);

        // 테스트 3: 0x00 (00000000) 전송
        $display("=== Test 3: Sending 0x00 ===");
        send_byte(8'h00);
        #(BIT_PERIOD * 2);

        // 테스트 4: 0xFF (11111111) 전송
        $display("=== Test 4: Sending 0xFF ===");
        send_byte(8'hFF);
        #(BIT_PERIOD * 2);

        // 테스트 5: 센서값 예시 0x96 (150) 전송
        $display("=== Test 5: Sending 0x96 (150) ===");
        send_byte(8'h96);
        #(BIT_PERIOD * 2);

        $display("=== All tests done ===");
        $stop;
    end

    // 결과 자동 확인
    always @(posedge data_valid) begin
        $display("[%0t ns] Received: 0x%02X (%0d)",
                 $time, data_out, data_out);
    end

endmodule
