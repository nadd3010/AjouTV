module fc_layer (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] attn_out [0:3][0:3],
    output reg  [15:0] score_normal,
    output reg  [15:0] score_abnormal,
    output reg         done
);

// TODO: Flatten  W_FC(2x16) 곱  정상/이상 score

endmodule
