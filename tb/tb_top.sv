`timescale 1ns / 1ps

module tb_top;

    localparam CLK_PERIOD = 20;       // 50MHz
    localparam BIT_PERIOD = 8680;     // 115200 baud

    reg        clk;
    reg        rst_n;
    reg        rx;
    wire       tx;
    wire [3:0] led;
    wire       buzzer;

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

    // UART 1바이트 전송 (PC → FPGA)
    task uart_send(input [7:0] data);
        integer i;
        begin
            rx = 1'b0;  // Start bit
            #(BIT_PERIOD);
            for (i = 0; i < 8; i = i + 1) begin
                rx = data[i];  // LSB first
                #(BIT_PERIOD);
            end
            rx = 1'b1;  // Stop bit
            #(BIT_PERIOD);
        end
    endtask

    // UART 1바이트 수신 (FPGA → PC)
    task uart_receive(output [7:0] data);
        integer i;
        begin
            @(negedge tx);       // Start bit 감지
            #(BIT_PERIOD / 2);   // Start bit 중앙
            for (i = 0; i < 8; i = i + 1) begin
                #(BIT_PERIOD);
                data[i] = tx;
            end
            #(BIT_PERIOD);       // Stop bit
        end
    endtask

    // 센서 데이터 16바이트 (4센서 × 4타임스텝)
    reg [7:0] sensor_data [0:15];
    // TX 수신 버퍼 9바이트
    reg [7:0] rx_buf [0:8];
    integer i;

    initial begin
        rst_n = 0;
        rx    = 1'b1;  // UART idle = HIGH
        #(CLK_PERIOD * 20);
        rst_n = 1;
        #(CLK_PERIOD * 20);

        // ===== 센서 데이터 설정 =====
        // 정상 패턴: 온도 100, 압력 100, 진동 100, 전류 100 × 4 타임스텝
        $display("=== Test 1: Normal pattern (all 100) ===");
        for (i = 0; i < 16; i = i + 1)
            sensor_data[i] = 8'd100;

        // 16바이트 UART 전송
        $display("[%0t] Sending 16 bytes via UART...", $time);
        for (i = 0; i < 16; i = i + 1) begin
            uart_send(sensor_data[i]);
            $display("[%0t] Sent byte[%0d] = %0d", $time, i, sensor_data[i]);
        end
        $display("[%0t] All 16 bytes sent", $time);

        // 처리 완료 대기
        $display("[%0t] Waiting for processing...", $time);

        // TX 출력 9바이트 수신
        $display("[%0t] Capturing TX output...", $time);
        for (i = 0; i < 9; i = i + 1) begin
            uart_receive(rx_buf[i]);
            $display("[%0t] TX byte[%0d] = 0x%02X (%0d)", $time, i, rx_buf[i], rx_buf[i]);
        end

        // 결과 출력
        $display("===== Results =====");
        $display("Label     : %0d (%s)", rx_buf[0], rx_buf[0] ? "ABNORMAL" : "NORMAL");
        $display("col_sum[0]: 0x%02X%02X", rx_buf[1], rx_buf[2]);
        $display("col_sum[1]: 0x%02X%02X", rx_buf[3], rx_buf[4]);
        $display("col_sum[2]: 0x%02X%02X", rx_buf[5], rx_buf[6]);
        $display("col_sum[3]: 0x%02X%02X", rx_buf[7], rx_buf[8]);
        $display("LED       : 4'b%04b", led);
        $display("Buzzer    : %0b", buzzer);

        #(CLK_PERIOD * 100);

        // ===== Test 2: 이상 패턴 =====
        $display("");
        $display("=== Test 2: Abnormal pattern (sensor0 = 250) ===");
        // 센서0만 비정상적으로 높은 값
        for (i = 0; i < 16; i = i + 1) begin
            if (i % 4 == 0)
                sensor_data[i] = 8'd250;  // 센서0: 이상
            else
                sensor_data[i] = 8'd100;  // 나머지: 정상
        end

        $display("[%0t] Sending 16 bytes via UART...", $time);
        for (i = 0; i < 16; i = i + 1) begin
            uart_send(sensor_data[i]);
            $display("[%0t] Sent byte[%0d] = %0d", $time, i, sensor_data[i]);
        end
        $display("[%0t] All 16 bytes sent", $time);

        $display("[%0t] Waiting for processing...", $time);

        $display("[%0t] Capturing TX output...", $time);
        for (i = 0; i < 9; i = i + 1) begin
            uart_receive(rx_buf[i]);
            $display("[%0t] TX byte[%0d] = 0x%02X (%0d)", $time, i, rx_buf[i], rx_buf[i]);
        end

        $display("===== Results =====");
        $display("Label     : %0d (%s)", rx_buf[0], rx_buf[0] ? "ABNORMAL" : "NORMAL");
        $display("col_sum[0]: 0x%02X%02X", rx_buf[1], rx_buf[2]);
        $display("col_sum[1]: 0x%02X%02X", rx_buf[3], rx_buf[4]);
        $display("col_sum[2]: 0x%02X%02X", rx_buf[5], rx_buf[6]);
        $display("col_sum[3]: 0x%02X%02X", rx_buf[7], rx_buf[8]);
        $display("LED       : 4'b%04b", led);
        $display("Buzzer    : %0b", buzzer);

        #(CLK_PERIOD * 100);
        $display("");
        $display("=== All integration tests done ===");
        $stop;
    end

    // FSM 상태 모니터
    always @(uut.u_main_sequencer.state) begin
        case (uut.u_main_sequencer.state)
            3'd0: $display("[%0t] FSM → IDLE", $time);
            3'd1: $display("[%0t] FSM → ATTENTION", $time);
            3'd2: $display("[%0t] FSM → CLASSIFY", $time);
            3'd3: $display("[%0t] FSM → OUTPUT", $time);
        endcase
    end

    // window_ready 모니터
    always @(posedge uut.u_sliding_window.window_ready)
        $display("[%0t] ★ window_ready!", $time);

endmodule
