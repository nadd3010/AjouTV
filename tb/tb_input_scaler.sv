`timescale 1ns / 1ps

module tb_input_scaler;

    localparam CLK_PERIOD = 20; // 50MHz

    reg         clk;
    reg         rst_n;
    reg  [7:0]  data_in;
    reg         data_valid;
    wire [15:0] scaled_out;
    wire        scaled_valid;

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

    task test_scale(input [7:0] in_val, input [15:0] expected);
        begin
            @(posedge clk);
            data_in    = in_val;
            data_valid = 1'b1;
            @(posedge clk);
            data_valid = 1'b0;
            @(posedge clk);

            if (scaled_out == expected)
                $display("[PASS] input=%0d -> output=0x%04X (expected 0x%04X)", in_val, scaled_out, expected);
            else
                $display("[FAIL] input=%0d -> output=0x%04X (expected 0x%04X)", in_val, scaled_out, expected);
        end
    endtask

    initial begin
        rst_n      = 0;
        data_in    = 0;
        data_valid = 0;
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);

        test_scale(8'd0,   16'h0000);  // 0 -> 0x0000
        test_scale(8'd1,   16'h0100);  // 1 -> 0x0100
        test_scale(8'd150, 16'h9600);  // 150 -> 0x9600
        test_scale(8'd255, 16'hFF00);  // 255 -> 0xFF00
        test_scale(8'd100, 16'h6400);  // 100 -> 0x6400

        $display("=== All tests done ===");
        $stop;
    end

endmodule
