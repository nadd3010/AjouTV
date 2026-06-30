module softmax_lut (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] score [0:3][0:3],
    output reg  [15:0] attn_weight [0:3][0:3],
    output reg         done
);

// TODO: LUT 기반 Softmax 근사

endmodule
