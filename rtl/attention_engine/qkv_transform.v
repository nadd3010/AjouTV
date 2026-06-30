module qkv_transform (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] x [0:3][0:3],
    output reg  [15:0] q [0:3][0:3],
    output reg  [15:0] k [0:3][0:3],
    output reg  [15:0] v [0:3][0:3],
    output reg         done
);

// TODO: X  W_Q, W_K, W_V  Q, K, V 생성

endmodule
