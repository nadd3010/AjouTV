`timescale 1ns / 1ps

module score_compute (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] q [0:3][0:3],
    input  wire [15:0] k [0:3][0:3],
    output reg  signed [15:0] score [0:3][0:3],
    output reg         done
);

typedef enum logic [1:0] {
        IDLE = 2'd0,
        CALC = 2'd1,
        DONE = 2'd2
    } state_t;

state_t state;

logic [2:0] row_idx; 
logic [2:0] col_idx;

logic signed [33:0] mac_val;    // 연산 결과
logic signed [33:0] shifted_mac;

// Q*K^T 내적 조합회로
always_comb begin
    mac_val = ($signed(q[row_idx][0]) * $signed(k[col_idx][0])) +
              ($signed(q[row_idx][1]) * $signed(k[col_idx][1])) +
              ($signed(q[row_idx][2]) * $signed(k[col_idx][2])) +
              ($signed(q[row_idx][3]) * $signed(k[col_idx][3]));
    shifted_mac = mac_val >>> 1;    // 1 right shift
end

always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            row_idx <= 0;
            col_idx <= 0;
            done <= 0;
            // 출력 행렬 초기화
            for (int i = 0; i < 4; i++) begin
                for (int j = 0; j < 4; j++) begin
                    score[i][j] <= 16'd0;
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
                    // Scaled Dot-Product: 내적 결과를 2로 나누기 (1-bit right shift)
                    // 연산 후 자릿수가 확장되더라도 Q8.8 형식을 유지하기 위해 [23:8] 비트 슬라이싱
                    score[row_idx][col_idx] <= shifted_mac[23:8];

                    // 4x4 행렬 순회 카운터 업데이트
                    if (col_idx == 3) begin
                        col_idx <= 0;
                        if (row_idx == 3) begin
                            state <= DONE; // 모든 연산 완료
                        end else begin
                            row_idx <= row_idx + 1;
                        end
                    end else begin
                        col_idx <= col_idx + 1;
                    end
                end

                DONE: begin
                    done <= 1;
                    state <= IDLE; // 다음 미니배치 연산을 위해 IDLE 대기
                end
            endcase
        end
    end

endmodule
