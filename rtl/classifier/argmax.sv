module argmax (
    input  wire [15:0] score_abnormal,
    output wire        label
);

// 임시 threshold 값: 0.5(Q8.8 포맷)
localparam [15:0] THRESHOLD = 16'h0080;

assign label = ($signed(score_abnormal) > $signed(THRESHOLD)) ? 1'b1 : 1'b0;

endmodule
