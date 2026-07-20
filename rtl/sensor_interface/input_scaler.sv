module input_scaler (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [7:0]  data_in,
    input  wire        data_valid,
    output reg  [15:0] scaled_out,
    output reg         scaled_valid
);

// TODO: 8비트  Q8.8 고정소수점 변환하기
    // 8비트 정수 → Q8.8 변환
    // 상위 8비트 = 정수부, 하위 8비트 = 0
    // 예: 150 (0x96) → 0x9600
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scaled_out   <= 16'd0;
            scaled_valid <= 1'b0;
        end else begin
            scaled_valid <= 1'b0;
            if (data_valid) begin
                scaled_out   <= {data_in, 8'b0};  // 8비트 왼쪽 시프트
                scaled_valid <= 1'b1;
            end
        end
    end

endmodule

