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

// TODO: X  W_Q, W_K, W_V  Q, K, V 생성

logic [15:0] weights_rom [0:47];   


// 가중치 행렬
logic signed [15:0] w_q [0:3][0:3];
logic signed [15:0] w_k [0:3][0:3];
logic signed [15:0] w_v [0:3][0:3];

integer i, j;

initial begin


    //*************** vivado 구울 때 경로 수정 ******************//
    // $readmemh("../mem/weights.mem", weights_rom);       // modelsim 용 경로
    $readmemh("weights.mem", weights_rom);       // vivado 용 경로
    


    for (i=0; i < 4; i = i + 1) begin
        for (j=0; j < 4; j = j + 1) begin
            w_q[i][j] = weights_rom[(i*4) + j];
            w_k[i][j] = weights_rom[16 + (i*4) + j];
            w_v[i][j] = weights_rom[32 + (i*4) + j];
        end
    end
end

// FSM 상태
localparam IDLE = 2'd0;
localparam CALC = 2'd1;
localparam DONE = 2'd2;

reg [1:0] state;
reg [1:0] row_idx;
reg [1:0] col_idx;

// 조합회로: 요소 한 칸 식 계산
reg signed [33:0] mac_q_elem;
reg signed [33:0] mac_k_elem;
reg signed [33:0] mac_v_elem;
integer dot;

always @(*) begin
    mac_q_elem = 34'sd0;
        mac_k_elem = 34'sd0;
        mac_v_elem = 34'sd0;
        
    for (dot = 0; dot < 4; dot = dot + 1) begin
        mac_q_elem = mac_q_elem + ($signed(x[row_idx][dot]) * $signed(w_q[dot][col_idx]));
        mac_k_elem = mac_k_elem + ($signed(x[row_idx][dot]) * $signed(w_k[dot][col_idx]));
        mac_v_elem = mac_v_elem + ($signed(x[row_idx][dot]) * $signed(w_v[dot][col_idx]));
    end
end

// 순차회로: 값 저장

integer out_i, out_j;

always @(posedge clk or negedge rst_n) begin 
    if (!rst_n) begin
        state <= IDLE;
        row_idx <=2'd0;
        col_idx <= 2'd0;
        done <= 1'd0;

        for(out_i = 0; out_i < 4; out_i = out_i +1) begin
            for (out_j = 0; out_j < 4; out_j = out_j + 1) begin
                q[out_i][out_j] <= 16'd0;
                k[out_i][out_j] <= 16'd0;
                v[out_i][out_j] <= 16'd0;
            end
        end
    end else begin
        case (state)
                IDLE: begin
                    done <= 1'b0;
                    row_idx <= 2'd0;
                    col_idx <= 2'd0;
                    if (start) state <= CALC; // 앞 모듈에서 start 신호가 오면 연산 시작
                end
                
                CALC: begin
                    //[23:8]로 슬라이싱하여 Q8.8 출력 (변경 가능)
                    q[row_idx][col_idx] <= mac_q_elem[23:8];
                    k[row_idx][col_idx] <= mac_k_elem[23:8];
                    v[row_idx][col_idx] <= mac_v_elem[23:8];

                    // 인덱스 이동 로직
                    if (col_idx == 2'd3) begin
                        col_idx <= 2'd0;
                        if (row_idx == 2'd3) begin
                            state <= DONE; 
                        end else begin
                            row_idx <= row_idx + 1'b1; 
                        end
                    end else begin
                        col_idx <= col_idx + 1'b1;
                    end
                end
                
                DONE: begin
                    done <= 1'b1; // 다음 모듈(score_compute)로 완료 신호 전달
                    state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end

        

endmodule
