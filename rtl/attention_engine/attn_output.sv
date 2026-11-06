
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

    // =========================================================
    // 1. FSM
    //    각 출력 원소를 3클럭(기존1클럭)에 걸쳐 계산 -> 4x4 전체 처리 48클럭
    // =========================================================
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

    // =========================================================
    // 2. 중간 연산 레지스터
    // =========================================================

    // Stage 1: 16x16 signed multiplication -> 32 bits
    logic signed [31:0] mul0_reg;
    logic signed [31:0] mul1_reg;
    logic signed [31:0] mul2_reg;
    logic signed [31:0] mul3_reg;

    // Stage 2: 2개의 곱셈 결과 합산 -> 33 bits
    logic signed [32:0] sum01_reg;
    logic signed [32:0] sum23_reg;

    // Stage 3: 4개 곱셈 결과의 최종 합산 -> 34 bits
    logic signed [33:0] mac_val;

    assign mac_val =
        $signed({sum01_reg[32], sum01_reg}) +
        $signed({sum23_reg[32], sum23_reg});

    // =========================================================
    // 3. 연산 데이터패스
    //
    // 중간 레지스터는 reset하지 않음.
    // MUL -> PAIR_SUM -> WRITE 순서가 보장되므로
    // 초기화되지 않은 값은 출력 계산에 사용되지 않음.
    //
    // DSP 내부 레지스터 최적화에 유리하도록 구성.
    // =========================================================
    always_ff @(posedge clk) begin
        if (rst_n) begin
            case (state)

                // ---------------------------------------------
                // Stage 1: 4개 곱셈 병렬 수행
                // ---------------------------------------------
                MUL: begin
                    mul0_reg <=
                        $signed(attn_weight[row_idx][0]) *
                        $signed(v[0][col_idx]);

                    mul1_reg <=
                        $signed(attn_weight[row_idx][1]) *
                        $signed(v[1][col_idx]);

                    mul2_reg <=
                        $signed(attn_weight[row_idx][2]) *
                        $signed(v[2][col_idx]);

                    mul3_reg <=
                        $signed(attn_weight[row_idx][3]) *
                        $signed(v[3][col_idx]);
                end

                // ---------------------------------------------
                // Stage 2: 2개씩 부분합 계산
                // 32 -> 33비트 부호 확장
                // ---------------------------------------------
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

    // =========================================================
    // 4. FSM 및 출력 제어
    // =========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= IDLE;
            row_idx <= 2'd0;
            col_idx <= 2'd0;
            done    <= 1'b0;

            for (int i = 0; i < 4; i++) begin
                for (int j = 0; j < 4; j++) begin
                    attn_out[i][j] <= 16'd0;
                end
            end

        end else begin

            done <= 1'b0;

            case (state)

                // ---------------------------------------------
                // IDLE: 연산 시작 대기
                // ---------------------------------------------
                IDLE: begin
                    row_idx <= 2'd0;
                    col_idx <= 2'd0;

                    if (start)
                        state <= MUL;
                end

                // ---------------------------------------------
                // Stage 1: 곱셈 결과 저장
                // ---------------------------------------------
                MUL: begin
                    state <= PAIR_SUM;
                end

                // ---------------------------------------------
                // Stage 2: 부분합 저장
                // ---------------------------------------------
                PAIR_SUM: begin
                    state <= WRITE;
                end

                // ---------------------------------------------
                // Stage 3: 최종 합산 및 Q8.8 출력 저장
                // ---------------------------------------------
                WRITE: begin

                    // 기존 Golden Model의 비트 슬라이싱 유지
                    attn_out[row_idx][col_idx] <= mac_val[23:8];

                    // 다음 출력 원소로 이동
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

                // ---------------------------------------------
                // DONE: 완료 펄스 1클럭 발생
                // ---------------------------------------------
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
