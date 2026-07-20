`timescale 1ns / 1ps

module softmax_lut (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] score [0:3][0:3],
    output reg  [15:0] attn_weight [0:3][0:3],
    output reg         done
);

    // 상태머신
    typedef enum logic [1:0] {
        IDLE = 2'b00,
        CALC = 2'b01,
        DONE = 2'b10
    } state_t;
    
    state_t state;
    
    logic [2:0] row_idx;
    logic [2:0] col_idx;

    logic signed [15:0] softmax_rom [0:255]; 


    initial begin
        $readmemh("../mem/softmax_rom.mem", softmax_rom);
    end


    // 인덱스 슬라이싱 및 Clipping (Saturation) 조합 회로
    wire signed [15:0] current_score = $signed(score[row_idx][col_idx]);
    logic [7:0]        rom_index;
    
    // 인덱스 슬라이싱 위치 "미정" 
    localparam SLICE_START = 4; // 시작 비트 위치 

    always_comb begin
        //Clipping(Saturation) 처리 
        if (current_score < 16'sd0) begin
            // 극단적인 음수 값일 경우 0번지로 강제 고정
            rom_index = 8'd0;
        end 
        else if (current_score > (16'sd255 << SLICE_START)) begin
            // 유효 범위를 초과하는 경우 255번지(최대 주소)로 강제 고정
            rom_index = 8'd255;
        end 
        else begin
            // 정상 유효 범위일 경우 8비트 슬라이싱 추출 [cite: 1560]
            rom_index = current_score[SLICE_START + 7 -: 8];
        end
    end

    // 순차 연산 FSM (Sequential Logic)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= IDLE;
            row_idx <= 0;
            col_idx <= 0;
            done    <= 0;
            
            for (int i = 0; i < 4; i++) begin
                for (int j = 0; j < 4; j++) begin
                    attn_weight[i][j] <= 16'd0;
                end
            end
        end 
        else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state   <= CALC;
                        row_idx <= 0;
                        col_idx <= 0;
                    end
                end
                
                CALC: begin
                    // 1클럭당 1개의 요소를 ROM에서 읽어와 출력 배열에 매핑
                    attn_weight[row_idx][col_idx] <= softmax_rom[rom_index];
                    
                    if (col_idx == 3) begin
                        col_idx <= 0;
                        if (row_idx == 3) begin
                            state <= DONE; // 16개 요소(4x4) 처리 완료
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