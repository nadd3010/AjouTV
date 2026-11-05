`timescale 1ns / 1ps

// ============================================================================
// tb_input_scaler (Q4.4 8비트 -> Q8.8 변환용으로 교체한 테스트)
//   입력은 부호 있는 Q4.4 (분해능 1/16, 범위 -8.0 ~ +7.9375).  Q8.8 원시값 = 부호 있는 바이트 x 16.
//   예전 기대값(예: 150 -> 0x9600)은 부호 없는 정수 변환이라 더 이상 맞지 않는다.
//   ModelSim 의 기존 규약에 맞춰 마지막에 $stop 을 쓴다.
// ============================================================================
module tb_input_scaler;

    localparam CLK_PERIOD = 10; // 100MHz 로 변경 예정 (기존 20 = 50MHz 였음)

    reg         clk;
    reg         rst_n;
    reg  [7:0]  data_in;
    reg         data_valid;
    wire [15:0] scaled_out;
    wire        scaled_valid;
    integer     fails;

    input_scaler uut (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_in),
        .data_valid(data_valid),
        .scaled_out(scaled_out),
        .scaled_valid(scaled_valid)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // 입력은 클럭 엣지 1ns 뒤에 바꿔서 DUT 와의 경쟁(race)을 없앤다
    task test_scale(input [7:0] in_val, input [15:0] expected);
        begin
            @(posedge clk); #1;
            data_in    = in_val;
            data_valid = 1'b1;
            @(posedge clk); #1;
            data_valid = 1'b0;
            @(posedge clk); #1;
            if (scaled_out === expected && scaled_valid === 1'b0)
                $display("[PASS] input=0x%02X -> output=0x%04X (expected 0x%04X)", in_val, scaled_out, expected);
            else begin
                $display("[FAIL] input=0x%02X -> output=0x%04X (expected 0x%04X)", in_val, scaled_out, expected);
                fails = fails + 1;
            end
        end
    endtask

    initial begin
        fails      = 0;
        rst_n      = 0;
        data_in    = 0;
        data_valid = 0;
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        test_scale(8'h00, 16'h0000);  //  0
        test_scale(8'h01, 16'h0010);  // +1/16
        test_scale(8'h10, 16'h0100);  // +1.0
        test_scale(8'hF0, 16'hFF00);  // -1.0
        test_scale(8'hFF, 16'hFFF0);  // -1/16
        test_scale(8'h7F, 16'h07F0);  // +7.9375 (최대)
        test_scale(8'h80, 16'hF800);  // -8.0 (최소)
        test_scale(8'h64, 16'h0640);  // +6.25
        test_scale(8'h96, 16'hF960);  // -6.625

        // scaled_valid 는 data_valid 다음 클럭에 1 이 되었다가 한 클럭 뒤에 0 으로 돌아와야 한다
        @(posedge clk); #1; data_in = 8'h10; data_valid = 1'b1;
        @(posedge clk); #1;
        if (scaled_valid === 1'b1) $display("[PASS] scaled_valid pulse");
        else begin $display("[FAIL] scaled_valid pulse"); fails = fails + 1; end
        data_valid = 1'b0;
        @(posedge clk); #1;
        if (scaled_valid === 1'b0) $display("[PASS] scaled_valid returns to 0");
        else begin $display("[FAIL] scaled_valid returns to 0"); fails = fails + 1; end

        if (fails == 0) $display("=== input_scaler: ALL PASS ===");
        else            $display("=== input_scaler: %0d FAIL ===", fails);
        $stop;
    end

endmodule