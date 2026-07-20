module fc_layer (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] attn_out [0:3][0:3],
    output reg  [15:0] score_abnormal,
    output reg         done
);

    // 1. 상태 머신 정의 (SystemVerilog)
    typedef enum logic [1:0] {
        IDLE = 2'b00,
        CALC = 2'b01,
        DONE = 2'b10
    } state_t;
    
    state_t state, next_state;

    // 2. 내부 변수 및 메모리 선언
    logic signed [15:0] w_fc [0:15]; // 1x16 가중치 ROM
    logic [3:0]         idx;         // 0~15 순회용 카운터
    logic signed [35:0] mac_val;     // 누산기 (오버플로우 방지용 36비트 확장)

    // 3. 가중치 초기화
    initial begin
        // FC 가중치 메모리 로드 (.mem 파일 경로는 통합 기준에 맞춤)
        $readmemh("../mem/fc_weights.mem", w_fc);
    end

    // 4. 상태 머신 및 순차 회로 (MAC 연산)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            idx <= 4'd0;
            mac_val <= 36'sd0;
            score_abnormal <= 16'd0;
            done <= 1'b0;
        end else begin
            state <= next_state;

            case (state)
                IDLE: begin
                    done <= 1'b0;
                    idx <= 4'd0;
                    mac_val <= 36'sd0;
                end
                CALC: begin
                    // 2차원 배열(attn_out)을 1차원 인덱스(idx)로 분할 접근 (4x4 = 16)
                    // idx[3:2]는 행(row, 0~3), idx[1:0]는 열(col, 0~3)에 해당
                    mac_val <= mac_val + ($signed(attn_out[idx[3:2]][idx[1:0]]) * w_fc[idx]);
                    idx <= idx + 1'b1;
                end
                DONE: begin
                    // Q8.8 포맷 복원 (비트 슬라이싱, 나중에 결과 보고 수정)
                    score_abnormal <= mac_val[23:8];
                    done <= 1'b1;
                end
            endcase
        end
    end

    // 5. 다음 상태 결정 로직
    always_comb begin
        next_state = state;
        case (state)
            IDLE: if (start) next_state = CALC;
            CALC: if (idx == 4'd15) next_state = DONE; // 16번 누적 연산 후 완료
            DONE: next_state = IDLE;
        endcase
    end

endmodule