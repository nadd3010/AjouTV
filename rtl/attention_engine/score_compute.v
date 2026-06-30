module score_compute (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] q [0:3][0:3],
    input  wire [15:0] k [0:3][0:3],
    output reg  [15:0] score [0:3][0:3],
    output reg         done
);

// TODO: Q  K^T 내적 후 1bit right shift (4)

endmodule
