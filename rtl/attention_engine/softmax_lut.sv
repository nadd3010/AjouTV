
`timescale 1ns / 1ps

module softmax_lut (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire signed [15:0] score [0:3][0:3],
    output reg  signed [15:0] attn_weight [0:3][0:3],
    output reg         done
);

    // =====================================================
    // 1. FSM
    // =====================================================
    typedef enum logic [2:0] {
        IDLE      = 3'd0,
        FIND_MAX  = 3'd1,
        EXP_INDEX = 3'd2,
        EXP_READ  = 3'd3,
        EXP_ACC   = 3'd4,
        CALC_INV  = 3'd5,
        CALC_MUL  = 3'd6,
        DONE      = 3'd7
    } state_t;

    state_t state;

    logic [2:0] row_idx;
    logic [2:0] col_idx;

    // =====================================================
    // 2. ROM
    // =====================================================

    // EXP ROM: 256 entries
    logic [15:0] exp_rom [0:255];

    initial begin
        // ModelSim 용
        // $readmemh("../mem/softmax_exp_rom.mem", exp_rom);

        // Vivado 용
        $readmemh("softmax_exp_rom.mem", exp_rom);
    end

    // Inverse ROM: 1025 entries
    logic [15:0] inv_rom [0:1024];

    initial begin
        // ModelSim 용
        // $readmemh("../mem/softmax_inv_rom.mem", inv_rom);

        // Vivado 용
        $readmemh("softmax_inv_rom.mem", inv_rom);
    end

    // =====================================================
    // 3. Max Finder
    // =====================================================
    logic signed [15:0] next_max_score [0:3];
    logic signed [15:0] max_score_reg  [0:3];

    always_comb begin
        for (int i = 0; i < 4; i++) begin
            next_max_score[i] = score[i][0];

            for (int j = 1; j < 4; j++) begin
                if (score[i][j] > next_max_score[i])
                    next_max_score[i] = score[i][j];
            end
        end
    end

    // =====================================================
    // 4. EXP Pipeline Registers
    // =====================================================

    // 각 행의 EXP 합계
    logic [17:0] exp_sum [0:3];

    // 각 원소의 EXP 값
    logic [15:0] exp_buffer [0:3][0:3];

    // Stage 1: ROM Index
    logic signed [15:0] shifted_score;
    logic [15:0] abs_score;
    logic [7:0] exp_rom_index;
    logic [7:0] exp_index_reg;

    // Stage 2: ROM Data
    logic [15:0] exp_val_reg;

    localparam integer SLICE_START = 3;

    assign shifted_score =
        score[row_idx][col_idx] - max_score_reg[row_idx];

    // 기존 EXP 인덱스 계산 방식 유지
    always_comb begin
        // 모든 경로에서 할당하여 latch 방지
        abs_score = $unsigned(-shifted_score);

        if (shifted_score <
            -($signed(16'd255) << SLICE_START)) begin

            exp_rom_index = 8'd0;

        end else begin

            exp_rom_index =
                8'd255 - (abs_score >> SLICE_START);

        end
    end

    // =====================================================
    // 5. Inverse ROM & Multiplication
    // =====================================================

    logic [10:0] inv_rom_index;
    logic [15:0] current_inv_val;

    assign inv_rom_index = exp_sum[row_idx][10:0];

    logic [31:0] mul_full;

    assign mul_full =
        exp_buffer[row_idx][col_idx] * current_inv_val;

    // =====================================================
    // 6. EXP Pipeline Datapath
    //
    // EXP_INDEX -> EXP_READ -> EXP_ACC
    //
    // 중간 레지스터에는 비동기 리셋을 사용하지 않음.
    // 유효한 결과는 FSM 순서에 따라 사용됨.
    // =====================================================
    always_ff @(posedge clk) begin
        if (rst_n) begin
            case (state)

                // -----------------------------------------
                // Stage 1: EXP 주소 저장
                // -----------------------------------------
                EXP_INDEX: begin
                    exp_index_reg <= exp_rom_index;
                end

                // -----------------------------------------
                // Stage 2: EXP ROM 데이터 저장
                // -----------------------------------------
                EXP_READ: begin
                    exp_val_reg <= exp_rom[exp_index_reg];
                end

                default: begin
                    // 레지스터 값 유지
                end

            endcase
        end
    end

    // =====================================================
    // 7. FSM & Output Registers
    // =====================================================
    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            state   <= IDLE;
            row_idx <= 3'd0;
            col_idx <= 3'd0;
            done    <= 1'b0;

            current_inv_val <= 16'd0;

            for (int i = 0; i < 4; i++) begin

                max_score_reg[i] <= 16'd0;
                exp_sum[i]       <= 18'd0;

                for (int j = 0; j < 4; j++) begin
                    attn_weight[i][j] <= 16'd0;
                    exp_buffer[i][j]  <= 16'd0;
                end
            end

        end else begin

            // 기본적으로 done은 0
            done <= 1'b0;

            case (state)

                // -----------------------------------------
                // IDLE
                // -----------------------------------------
                IDLE: begin
                    if (start) begin

                        row_idx <= 3'd0;
                        col_idx <= 3'd0;

                        // 각 행의 합 초기화
                        for (int i = 0; i < 4; i++) begin
                            exp_sum[i] <= 18'd0;
                        end

                        state <= FIND_MAX;
                    end
                end

                // -----------------------------------------
                // FIND_MAX: 각 행의 최댓값 저장
                // -----------------------------------------
                FIND_MAX: begin

                    for (int i = 0; i < 4; i++) begin
                        max_score_reg[i] <= next_max_score[i];
                    end

                    row_idx <= 3'd0;
                    col_idx <= 3'd0;

                    state <= EXP_INDEX;
                end

                // -----------------------------------------
                // Stage 1: EXP 인덱스 계산
                // -----------------------------------------
                EXP_INDEX: begin
                    // Datapath에서 exp_index_reg 저장
                    state <= EXP_READ;
                end

                // -----------------------------------------
                // Stage 2: EXP ROM 읽기
                // -----------------------------------------
                EXP_READ: begin
                    // Datapath에서 exp_val_reg 저장
                    state <= EXP_ACC;
                end

                // -----------------------------------------
                // Stage 3: EXP 저장 및 행별 합계 누적
                // -----------------------------------------
                EXP_ACC: begin

                    exp_buffer[row_idx][col_idx]
                        <= exp_val_reg;

                    exp_sum[row_idx]
                        <= exp_sum[row_idx] + exp_val_reg;

                    // 다음 원소로 이동
                    if (col_idx == 3'd3) begin

                        col_idx <= 3'd0;

                        if (row_idx == 3'd3) begin

                            // 모든 EXP 연산 완료
                            row_idx <= 3'd0;
                            state <= CALC_INV;

                        end else begin

                            row_idx <= row_idx + 3'd1;
                            state <= EXP_INDEX;

                        end

                    end else begin

                        col_idx <= col_idx + 3'd1;
                        state <= EXP_INDEX;

                    end
                end

                // -----------------------------------------
                // CALC_INV: 행별 역수 ROM 읽기
                // -----------------------------------------
                CALC_INV: begin

                    current_inv_val <=
                        inv_rom[inv_rom_index];

                    col_idx <= 3'd0;
                    state <= CALC_MUL;
                end

                // -----------------------------------------
                // CALC_MUL: EXP * Inverse
                // -----------------------------------------
                CALC_MUL: begin

                    // Q8.8 * Q8.8 = Q16.16
                    // 결과 [23:8] 슬라이싱 유지
                    attn_weight[row_idx][col_idx]
                        <= mul_full[23:8];

                    if (col_idx == 3'd3) begin

                        if (row_idx == 3'd3) begin

                            state <= DONE;

                        end else begin

                            row_idx <= row_idx + 3'd1;
                            state <= CALC_INV;

                        end

                    end else begin

                        col_idx <= col_idx + 3'd1;

                    end
                end

                // -----------------------------------------
                // DONE: 완료 펄스 발생
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
