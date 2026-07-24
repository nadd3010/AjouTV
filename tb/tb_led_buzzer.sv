`timescale 1ns / 1ps

module tb_led_buzzer;

    localparam CLK_PERIOD = 20;

    reg        clk;
    reg        rst_n;
    reg        label;
    reg  [1:0] cause_index;
    wire [3:0] led;
    wire       buzzer;

    led_controller u_led (
        .label(label),
        .cause_index(cause_index),
        .led(led)
    );

    buzzer_driver u_buzz (
        .clk(clk),
        .rst_n(rst_n),
        .label(label),
        .buzzer(buzzer)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    integer pass_cnt;
    integer fail_cnt;

    task check_led(input [3:0] expected);
        begin
            if (led === expected) begin
                $display("[PASS] led = 4'b%04b (expected 4'b%04b)", led, expected);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] led = 4'b%04b (expected 4'b%04b)", led, expected);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    task check_buzzer(input logic expected);
        begin
            if (buzzer === expected) begin
                $display("[PASS] buzzer = %0b (expected %0b)", buzzer, expected);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] buzzer = %0b (expected %0b)", buzzer, expected);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    initial begin
        pass_cnt    = 0;
        fail_cnt    = 0;
        rst_n       = 0;
        label       = 0;
        cause_index = 2'd0;
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        // ===== Test 1: 정상 (label=0) =====
        $display("=== Test 1: Normal (label=0) ===");
        label       = 1'b0;
        cause_index = 2'd0;
        #(CLK_PERIOD * 2);
        check_led(4'b0000);
        check_buzzer(1'b0);

        // ===== Test 2: 이상, 센서0 원인 (cause_index=0) =====
        $display("=== Test 2: Abnormal, cause=0 ===");
        @(posedge clk);
        label       = 1'b1;
        cause_index = 2'd0;
        #(CLK_PERIOD * 2);
        check_led(4'b0001);
        check_buzzer(1'b1);

        // ===== Test 3: 이상, 센서1 원인 (cause_index=1) =====
        $display("=== Test 3: Abnormal, cause=1 ===");
        cause_index = 2'd1;
        #(CLK_PERIOD * 2);
        check_led(4'b0010);

        // ===== Test 4: 이상, 센서2 원인 (cause_index=2) =====
        $display("=== Test 4: Abnormal, cause=2 ===");
        cause_index = 2'd2;
        #(CLK_PERIOD * 2);
        check_led(4'b0100);

        // ===== Test 5: 이상, 센서3 원인 (cause_index=3) =====
        $display("=== Test 5: Abnormal, cause=3 ===");
        cause_index = 2'd3;
        #(CLK_PERIOD * 2);
        check_led(4'b1000);

        // ===== Test 6: 다시 정상 → LED OFF, 부저 유지 =====
        $display("=== Test 6: Back to normal ===");
        @(posedge clk);
        label = 1'b0;
        #(CLK_PERIOD * 2);
        check_led(4'b0000);
        check_buzzer(1'b1);  // 부저는 0.5초 타이머 만료까지 유지

        // ===== 결과 요약 =====
        $display("=================================");
        $display("Total: %0d PASS, %0d FAIL", pass_cnt, fail_cnt);
        $display("=================================");
        $stop;
    end

endmodule
