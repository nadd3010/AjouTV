`timescale 1ns / 1ps

module attn_output (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire signed [15:0] attn_weight [0:3][0:3],
    input  wire [15:0] v [0:3][0:3],
    output reg  [15:0] attn_out [0:3][0:3],
    output reg         done
);

// 1. 상태 머신 (FSM) 정의
typedef enum logic [1:0] {
    IDLE = 2'b00,
    CALC = 2'b01,
    DONE = 2'b10
} state_t;

state_t state;

// 2. 내부 변수 선언 (SystemVerilog logic 사용)
logic [2:0] row_idx;
logic [2:0] col_idx;
logic signed [33:0] mac_val; // 16x16=32비트 + 4번 누적(2비트) = 34비트 오버플로우 방지

// 3. 조합 회로: 4개의 요소 병렬 내적 연산 (4 DSPs)
always_comb begin
    // 외부 포트의 부호 없음을 내부에서 $signed()로 명시적 캐스팅
    mac_val = ( $signed(attn_weight[row_idx][0]) * $signed(v[0][col_idx]) ) +
              ( $signed(attn_weight[row_idx][1]) * $signed(v[1][col_idx]) ) +
              ( $signed(attn_weight[row_idx][2]) * $signed(v[2][col_idx]) ) +
              ( $signed(attn_weight[row_idx][3]) * $signed(v[3][col_idx]) );
end

// 4. 순차 회로: FSM 제어 및 결과 저장
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        row_idx <= 0;
        col_idx <= 0;
        done <= 0;
        // 출력 배열 초기화 (X값 전파 방지)
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                attn_out[i][j] <= 16'd0;
            end
        end
    end else begin
        case (state)
            IDLE: begin
                done <= 0;
                row_idx <= 0;
                col_idx <= 0;
                if (start) state <= CALC;
            end

            CALC: begin
                // 곱셈 누적으로 확장된 비트 폭에서 [23:8]을 잘라 Q8.8 포맷 유지
                attn_out[row_idx][col_idx] <= mac_val[23:8];

                // 인덱스 카운터 제어 (4x4 행렬 순회)
                if (col_idx == 3) begin
                    col_idx <= 0;
                    if (row_idx == 3) begin
                        state <= DONE; // 16개 요소 모두 계산 완료
                    end else begin
                        row_idx <= row_idx + 1;
                    end
                end else begin
                    col_idx <= col_idx + 1;
                end
            end

            DONE: begin
                done <= 1;
                state <= IDLE;
            end
            
            default: state <= IDLE;
        endcase
    end
end

endmodule