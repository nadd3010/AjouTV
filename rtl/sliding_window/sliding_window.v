module sliding_window (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] sensor_data,
    input  wire        data_valid,
    output reg  [15:0] window [0:3][0:3],
    output reg         window_ready
);

// TODO: 시프트 레지스터 기반 4x4 행렬 구성

endmodule
