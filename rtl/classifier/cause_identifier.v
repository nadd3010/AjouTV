module cause_identifier (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] attn_weight [0:3][0:3],
    output reg  [17:0] col_sum [0:3],
    output reg  [1:0]  cause_index
);

// TODO: 센서별 Attention Score 합산  최대 기여 센서 식별

endmodule
