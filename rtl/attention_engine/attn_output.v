module attn_output (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] attn_weight [0:3][0:3],
    input  wire [15:0] v [0:3][0:3],
    output reg  [15:0] attn_out [0:3][0:3],
    output reg         done
);

// TODO: Softmax 결과  V  Attention 출력

endmodule
