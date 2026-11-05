// ============================================================================
// input_scaler (Q4.4 8비트 -> Q8.8 16비트)
//
// PC 가 센서별로 표준화한 값 x_std 를 q = clip(round(x_std * 16), -128, 127) 로 바꿔 부호 있는 8비트(Q4.4)로 보낸다.
// 이 모듈은 그것을 Q8.8 로 복원한다: Q8.8 원시값 = q * 16 = {부호 확장, q, 4'b0000}.
//   예) 0x10 (+1.0)  -> 0x0100,  0xF0 (-1.0) -> 0xFF00,  0x7F (+7.9375) -> 0x07F0,  0x80 (-8.0) -> 0xF800
//
// 변경 이유: 예전에는 {data_in, 8'b0} (0~255 의 부호 없는 정수)로 바꿔서, 모델이 기대하는 표준화된 값(약 -6 ~ +8)을 표현할 수 없었다.
// 입력 인코딩 비교(exp_input_encoding.py, Q8.8 골든 모델): 16비트 대비 AUC 하락 0.0019, 판정 일치 0.990.
// ============================================================================
module input_scaler (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [7:0]  data_in,       // 부호 있는 Q4.4 (두 의 보수), 분해능 1/16, 범위 -8.0 ~ +7.9375
    input  wire        data_valid,
    output reg  [15:0] scaled_out,    // Q8.8
    output reg         scaled_valid
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scaled_out   <= 16'd0;
            scaled_valid <= 1'b0;
        end else begin
            scaled_valid <= 1'b0;
            if (data_valid) begin
                scaled_out   <= {{4{data_in[7]}}, data_in, 4'b0000};   // 부호 확장 후 4비트 왼쪽 시프트
                scaled_valid <= 1'b1;
            end
        end
    end

endmodule