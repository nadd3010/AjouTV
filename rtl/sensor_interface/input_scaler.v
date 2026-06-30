module input_scaler (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [7:0]  data_in,
    input  wire        data_valid,
    output reg  [15:0] scaled_out,
    output reg         scaled_valid
);

// TODO: 8비트  Q8.8 고정소수점 변환하기

endmodule
