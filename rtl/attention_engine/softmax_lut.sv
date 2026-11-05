`timescale 1ns / 1ps

module softmax_lut (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire signed [15:0] score [0:3][0:3],
    output reg  signed [15:0] attn_weight [0:3][0:3],
    output reg         done
);

    // 1. 상태 머신 (5단계 파이프라인으로 확장)
    typedef enum logic [2:0] {
        IDLE        = 3'b000,
        FIND_MAX    = 3'b001,  // 1. 최댓값 탐색 및 래치
        CALC_EXP    = 3'b010,  // 2. e^(x-max) 계산 및 분모 합산
        CALC_INV    = 3'b011,  // 3. 분모의 역수(1/Sum) ROM 읽기
        CALC_MUL    = 3'b100,  // 4. DSP로 (분자 * 역수) 곱셈하여 최종 확률 도출
        DONE        = 3'b101
    } state_t;
    
    state_t state;
    
    logic [2:0] row_idx;
    logic [2:0] col_idx;

    // 2. ROM 선언 (EXP ROM & INV ROM)
    // 1) 분자 계산용 (e^(x-max)) ROM
    logic [15:0] exp_rom [0:255]; 
    initial begin


    //*************** vivado 구울 때 경로 수정 ******************//
        $readmemh("../mem/softmax_exp_rom.mem", exp_rom);    // modelsim 경로 
        // $readmemh("softmax_exp_rom.mem", exp_rom);       // vivado 경로

    
    end

    // 2) 분모 역수 계산용 (1/Sum) ROM
    // [수정 3] softmax_inv_rom.mem 은 1025칸이다 (make_inverse_rom.py: inv[i] = 65536 / i, i = 합의 Q8.8 원시값 0~1024).
    //          합은 최대 4 x 0x0100 = 0x0400 이므로 11비트 주소(0~1024)로 충분하다.
    logic [15:0] inv_rom [0:1024]; 
    initial begin


    //*************** vivado 구울 때 경로 수정 ******************//
        $readmemh("../mem/softmax_inv_rom.mem", inv_rom);   // modelsim 경로
        // $readmemh("softmax_inv_rom.mem", inv_rom);   // vivado 경로
    
    
    end


    // 3. Max Finder 및 EXP 연산 변수
    logic signed [15:0] next_max_score [0:3];
    logic signed [15:0] max_score_reg  [0:3]; 
    
    // 행별 누적 합(Sum)을 담을 배열 (오버플로우 방지를 위해 18비트로 선언)
    logic [17:0] exp_sum [0:3];
    
    // 분자값(e^x)들을 임시로 보관할 버퍼
    logic [15:0] exp_buffer [0:3][0:3];

    // 순수 조합 회로: 항상 4개 행의 최댓값을 계산해두고 대기함
    always_comb begin
        for (int i = 0; i < 4; i++) begin
            next_max_score[i] = score[i][0];
            for (int j = 1; j < 4; j++) begin
                if (score[i][j] > next_max_score[i]) begin
                    next_max_score[i] = score[i][j];
                end
            end
        end
    end

    // Shifted Score 및 EXP ROM 인덱스
    wire signed [15:0] shifted_score = score[row_idx][col_idx] - max_score_reg[row_idx];
    logic [7:0]  exp_rom_index;
    logic [15:0] abs_score;
    // [수정 4] softmax_exp_rom.mem 은 [-8, 0] 을 256칸으로 나눈 것(make_lut.py)이라 한 칸이 약 1/32 = 8 원시값.
    //          따라서 SLICE_START = 3 (예전 4 는 한 칸을 1/16 로 읽어 softmax 가 2배 평평해졌다)
    localparam SLICE_START = 3; 

    always_comb begin
        if (shifted_score < -($signed(16'd255) << SLICE_START)) begin
            exp_rom_index = 8'd0;
        end 
        else begin
            abs_score = $unsigned(-shifted_score);
            exp_rom_index = 8'd255 - (abs_score >> SLICE_START);
        end
    end

    // 4. 역수 및 DSP 곱셈 연산 변수
    logic [10:0] inv_rom_index;
    logic [15:0] current_inv_val; // 읽어온 역수 값
    
    // [수정 3] 합(Sum)의 Q8.8 원시값을 그대로 11비트 주소로 사용 (예전에는 [15:8], 즉 정수부 0~4만 사용)
    always_comb begin
        inv_rom_index = exp_sum[row_idx][10:0];
    end

    // [수정 2] 곱을 32비트로 계산한다. (예전: 16비트 문맥에서 곱이 계산되어 65536 이상이면 >>8 전에 잘림)
    logic [31:0] mul_full;
    always_comb begin
        mul_full = exp_buffer[row_idx][col_idx] * current_inv_val;
    end

    // 5. 순차 연산 fsm_state
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= IDLE;
            row_idx <= 0;
            col_idx <= 0;
            done    <= 0;
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
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        // 변수 초기화
                        for (int i = 0; i < 4; i++) begin
                            exp_sum[i] <= 18'd0;
                        end
                        state <= FIND_MAX;
                    end
                end
                
                // ---------------------------------------------------------
                // 1단계: 행별 최댓값 찾기 (1클럭 소요)
                // ---------------------------------------------------------
                FIND_MAX: begin
                    for (int i = 0; i < 4; i++) begin
                        max_score_reg[i] <= next_max_score[i];
                    end
                    row_idx <= 0;
                    col_idx <= 0;
                    state   <= CALC_EXP; 
                end
                
                // ---------------------------------------------------------
                // 2단계: EXP 연산 및 합계 누적 (16클럭 소요)
                // ---------------------------------------------------------
                CALC_EXP: begin
                    // ROM에서 읽은 e^x 값을 버퍼에 저장하고, 동시에 행별 분모(Sum) 누적
                    exp_buffer[row_idx][col_idx] <= exp_rom[exp_rom_index];
                    exp_sum[row_idx] <= exp_sum[row_idx] + exp_rom[exp_rom_index];
                    
                    if (col_idx == 3) begin
                        col_idx <= 0;
                        if (row_idx == 3) begin
                            row_idx <= 0;
                            state <= CALC_INV; // 16개 연산 완료 후 다음 단계로
                        end else begin
                            row_idx <= row_idx + 1;
                        end
                    end else begin
                        col_idx <= col_idx + 1;
                    end
                end

                // ---------------------------------------------------------
                // 3단계: 역수 ROM 읽기 (4클럭 소요 - 행별로 1번씩만 읽음)
                // ---------------------------------------------------------
                CALC_INV: begin
                    // 각 행의 누적 합(Sum)을 주소로 하여 역수(1/Sum)를 읽어옵니다.
                    current_inv_val <= inv_rom[inv_rom_index];
                    col_idx <= 0; // 곱셈을 위해 열 인덱스 초기화
                    state <= CALC_MUL;
                end

                // ---------------------------------------------------------
                // 4단계: 최종 곱셈 (DSP 연산, 16클럭 소요)
                // ---------------------------------------------------------
                CALC_MUL: begin
                    // 💡 [DSP 사용] 버퍼에 있던 분자값 * 읽어온 역수값 (Q8.8 * Q8.8 = Q16.16)
                    // 곱셈 결과를 Q8.8로 다시 맞추기 위해 >> 8 (비트 시프트) 수행
                    attn_weight[row_idx][col_idx] <= mul_full[23:8];   // = (곱 >> 8) 의 하위 16비트
                    
                    if (col_idx == 3) begin
                        if (row_idx == 3) begin
                            state <= DONE; // 모든 행의 곱셈 완료
                        end else begin
                            row_idx <= row_idx + 1;
                            state <= CALC_INV; // 다음 행의 역수를 읽으러 돌아감
                        end
                    end else begin
                        col_idx <= col_idx + 1;
                    end
                end
                
                // ---------------------------------------------------------
                DONE: begin
                    done  <= 1;
                    state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule