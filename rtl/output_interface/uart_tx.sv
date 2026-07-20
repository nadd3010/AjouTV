module uart_tx (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire        label,
    input  wire [17:0] col_sum [0:3],
    output reg         tx,
    output reg         done
);

// TODO: 9바이트 프로토콜 (label + col_sum[0~3] 각 2바이트) 전송

endmodule
