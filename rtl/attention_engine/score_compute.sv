
`timescale 1ns / 1ps

module score_compute (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] q [0:3][0:3],
    input  wire [15:0] k [0:3][0:3],
    output reg signed [15:0] score [0:3][0:3],
    output reg         done
);

    // =====================================================
    // 1. FSM 정의
    // =====================================================
    typedef enum logic [2:0] {
        IDLE     = 3'd0,
        MUL      = 3'd1,
        PAIR_SUM = 3'd2,
        WRITE    = 3'd3,
        DONE     = 3'd4
    } state_t;

    state_t state;

    logic [1:0] row_idx;
    logic [1:0] col_idx;

    // =====================================================
    // 2. Pipeline 연산 레지스터
    // =====================================================

    // Stage 1: 16x16 signed multiplication = 32 bits
    logic signed [31:0] mul0_reg;
    logic signed [31:0] mul1_reg;
    logic signed [31:0] mul2_reg;
    logic signed [31:0] mul3_reg;

    // Stage 2: 2개씩 합산 = 33 bits
    logic signed [32:0] sum01_reg;
    logic signed [32:0] sum23_reg;

    // Stage 3: 최종 4개 누적 = 34 bits
    logic signed [33:0] mac_val;
    logic signed [33:0] shifted_mac;

    assign mac_val =
        $signed({sum01_reg[32], sum01_reg}) +
        $signed({sum23_reg[32], sum23_reg});

    // Scaled Dot-Product: sqrt(4) = 2
    assign shifted_mac = mac_val >>> 1;

    // =====================================================
    // 3. Pipeline 데이터패스
    //
    // 중간 연산 레지스터는 비동기 리셋을 사용하지 않음.
    // DSP 내부 레지스터 최적화 가능성을 높이기 위함.
    //
    // MUL -> PAIR_SUM -> WRITE 순서로 동작하므로
    // 유효하지 않은 초기 중간값은 출력에 사용되지 않음.
    // =====================================================
    always_ff @(posedge clk) begin
        if (rst_n) begin
            case (state)

                // -----------------------------------------
                // Stage 1: 4개 곱셈 병렬 연산
                // -----------------------------------------
                MUL: begin
                    mul0_reg <=
                        $signed(q[row_idx][0]) *
                        $signed(k[col_idx][0]);

                    mul1_reg <=
                        $signed(q[row_idx][1]) *
                        $signed(k[col_idx][1]);

                    mul2_reg <=
                        $signed(q[row_idx][2]) *
                        $signed(k[col_idx][2]);

                    mul3_reg <=
                        $signed(q[row_idx][3]) *
                        $signed(k[col_idx][3]);
                end

                // -----------------------------------------
                // Stage 2: 2개씩 부분합 계산
                // 32비트 값을 33비트로 부호 확장
                // -----------------------------------------
                PAIR_SUM: begin
                    sum01_reg <=
                        $signed({mul0_reg[31], mul0_reg}) +
                        $signed({mul1_reg[31], mul1_reg});

                    sum23_reg <=
                        $signed({mul2_reg[31], mul2_reg}) +
                        $signed({mul3_reg[31], mul3_reg});
                end

                default: begin
                    // 중간 결과 유지
                end

            endcase
        end
    end

    // =====================================================
    // 4. FSM 및 Score 출력 제어
    // =====================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= IDLE;
            row_idx <= 2'd0;
            col_idx <= 2'd0;
            done    <= 1'b0;

            // 출력 행렬 초기화
            for (int i = 0; i < 4; i++) begin
                for (int j = 0; j < 4; j++) begin
                    score[i][j] <= 16'd0;
                end
            end

        end else begin
            done <= 1'b0;

            case (state)

                // -----------------------------------------
                // IDLE: 시작 신호 대기
                // -----------------------------------------
                IDLE: begin
                    row_idx <= 2'd0;
                    col_idx <= 2'd0;

                    if (start)
                        state <= MUL;
                end

                // -----------------------------------------
                // Stage 1: 곱셈 결과 저장
                // -----------------------------------------
                MUL: begin
                    state <= PAIR_SUM;
                end

                // -----------------------------------------
                // Stage 2: 부분합 저장
                // -----------------------------------------
                PAIR_SUM: begin
                    state <= WRITE;
                end

                // -----------------------------------------
                // Stage 3: 최종 누적 및 Score 저장
                // -----------------------------------------
                WRITE: begin

                    // 기존 Scaled Dot-Product 및
                    // Q8.8 슬라이싱 방식 유지
                    score[row_idx][col_idx] <=
                        shifted_mac[23:8];

                    // 다음 원소로 이동
                    if (col_idx == 2'd3) begin
                        col_idx <= 2'd0;

                        if (row_idx == 2'd3) begin
                            state <= DONE;
                        end else begin
                            row_idx <= row_idx + 2'd1;
                            state <= MUL;
                        end

                    end else begin
                        col_idx <= col_idx + 2'd1;
                        state <= MUL;
                    end
                end

                // -----------------------------------------
                // DONE: 완료 신호 1클럭 발생
                // -----------------------------------------
                DONE: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end

                default: begin
                    state <= IDLE;
                end

            endcase
        end
    end

endmodule
