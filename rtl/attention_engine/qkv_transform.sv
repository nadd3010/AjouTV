
`timescale 1ns / 1ps

module qkv_transform (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] x [0:3][0:3],
    output reg  [15:0] q [0:3][0:3],
    output reg  [15:0] k [0:3][0:3],
    output reg  [15:0] v [0:3][0:3],
    output reg         done
);

    // =====================================================
    // 1. Weight ROM
    // =====================================================
    logic [15:0] weights_rom [0:47];

    logic signed [15:0] w_q [0:3][0:3];
    logic signed [15:0] w_k [0:3][0:3];
    logic signed [15:0] w_v [0:3][0:3];

    integer i, j;

    initial begin
        // ModelSim용
        // $readmemh("../mem/weights.mem", weights_rom);

        // Vivado용 (프로젝트 구성에 맞게 변경)
        $readmemh("weights.mem", weights_rom);

        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                w_q[i][j] = weights_rom[(i*4) + j];
                w_k[i][j] = weights_rom[16 + (i*4) + j];
                w_v[i][j] = weights_rom[32 + (i*4) + j];
            end
        end
    end

    // =====================================================
    // 2. FSM
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
    // 3. Pipeline Registers
    // =====================================================

    // Stage 1: 16x16 signed multiplication
    // Q/K/V 각각 4개 곱셈 결과
    logic signed [31:0] mul_q_reg [0:3];
    logic signed [31:0] mul_k_reg [0:3];
    logic signed [31:0] mul_v_reg [0:3];

    // Stage 2: Pair-wise Sum (33 bits)
    logic signed [32:0] sum_q01_reg;
    logic signed [32:0] sum_q23_reg;

    logic signed [32:0] sum_k01_reg;
    logic signed [32:0] sum_k23_reg;

    logic signed [32:0] sum_v01_reg;
    logic signed [32:0] sum_v23_reg;

    // Stage 3: Final Accumulation (34 bits)
    logic signed [33:0] mac_q_elem;
    logic signed [33:0] mac_k_elem;
    logic signed [33:0] mac_v_elem;

    assign mac_q_elem =
        $signed({sum_q01_reg[32], sum_q01_reg}) +
        $signed({sum_q23_reg[32], sum_q23_reg});

    assign mac_k_elem =
        $signed({sum_k01_reg[32], sum_k01_reg}) +
        $signed({sum_k23_reg[32], sum_k23_reg});

    assign mac_v_elem =
        $signed({sum_v01_reg[32], sum_v01_reg}) +
        $signed({sum_v23_reg[32], sum_v23_reg});

    // =====================================================
    // 4. Pipeline Datapath
    //
    // 중간 레지스터에는 비동기 Reset을 사용하지 않음.
    // FSM에 의해 MUL -> PAIR_SUM -> WRITE 순서가 보장됨.
    // =====================================================
    always_ff @(posedge clk) begin
        if (rst_n) begin

            case (state)

                // -----------------------------------------
                // Stage 1: Multiplication
                // Q/K/V 각 4개 곱셈 병렬 수행
                // -----------------------------------------
                MUL: begin
                    for (int d = 0; d < 4; d++) begin

                        mul_q_reg[d] <=
                            $signed(x[row_idx][d]) *
                            $signed(w_q[d][col_idx]);

                        mul_k_reg[d] <=
                            $signed(x[row_idx][d]) *
                            $signed(w_k[d][col_idx]);

                        mul_v_reg[d] <=
                            $signed(x[row_idx][d]) *
                            $signed(w_v[d][col_idx]);

                    end
                end

                // -----------------------------------------
                // Stage 2: Pair-wise Addition
                // 각 32비트 결과를 33비트로 부호 확장
                // -----------------------------------------
                PAIR_SUM: begin

                    sum_q01_reg <=
                        $signed({mul_q_reg[0][31], mul_q_reg[0]}) +
                        $signed({mul_q_reg[1][31], mul_q_reg[1]});

                    sum_q23_reg <=
                        $signed({mul_q_reg[2][31], mul_q_reg[2]}) +
                        $signed({mul_q_reg[3][31], mul_q_reg[3]});

                    sum_k01_reg <=
                        $signed({mul_k_reg[0][31], mul_k_reg[0]}) +
                        $signed({mul_k_reg[1][31], mul_k_reg[1]});

                    sum_k23_reg <=
                        $signed({mul_k_reg[2][31], mul_k_reg[2]}) +
                        $signed({mul_k_reg[3][31], mul_k_reg[3]});

                    sum_v01_reg <=
                        $signed({mul_v_reg[0][31], mul_v_reg[0]}) +
                        $signed({mul_v_reg[1][31], mul_v_reg[1]});

                    sum_v23_reg <=
                        $signed({mul_v_reg[2][31], mul_v_reg[2]}) +
                        $signed({mul_v_reg[3][31], mul_v_reg[3]});

                end

                default: begin
                    // 중간 결과 유지
                end

            endcase
        end
    end

    // =====================================================
    // 5. FSM and Output Control
    // =====================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin

            state   <= IDLE;
            row_idx <= 2'd0;
            col_idx <= 2'd0;
            done    <= 1'b0;

            // Q/K/V 출력 행렬 초기화
            for (int r = 0; r < 4; r++) begin
                for (int c = 0; c < 4; c++) begin
                    q[r][c] <= 16'd0;
                    k[r][c] <= 16'd0;
                    v[r][c] <= 16'd0;
                end
            end

        end else begin

            // done은 기본적으로 0
            done <= 1'b0;

            case (state)

                // -----------------------------------------
                // IDLE
                // -----------------------------------------
                IDLE: begin
                    row_idx <= 2'd0;
                    col_idx <= 2'd0;

                    if (start)
                        state <= MUL;
                end

                // -----------------------------------------
                // Stage 1: Multiplication
                // -----------------------------------------
                MUL: begin
                    state <= PAIR_SUM;
                end

                // -----------------------------------------
                // Stage 2: Pair-wise Sum
                // -----------------------------------------
                PAIR_SUM: begin
                    state <= WRITE;
                end

                // -----------------------------------------
                // Stage 3: Final Sum and Output
                // -----------------------------------------
                WRITE: begin

                    // 기존 Q8.8 슬라이싱 유지
                    q[row_idx][col_idx] <= mac_q_elem[23:8];
                    k[row_idx][col_idx] <= mac_k_elem[23:8];
                    v[row_idx][col_idx] <= mac_v_elem[23:8];

                    // 다음 행렬 원소로 이동
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
