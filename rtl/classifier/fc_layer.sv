`timescale 1ns / 1ps

// ============================================================================
// fc_layer (res_square 구조)
//
//   f[i]  = attn_out[i] + x[i]                 (Q8.8 덧셈, 16비트 wrap)   <- 잔차 연결
//   sq[i] = (f[i] * f[i])[23:8]                (Q8.8)                      <- 제곱
//   z     = ( sum_i sq[i] * w_fc[i] )[23:8] + b    (Q8.8, 16비트 wrap)     <- 가중합 + 바이어스
//
// main 대비 변경점
//   [수정 1] w_fc 를 signed 로 선언. (예전: unsigned 라서 `$signed(attn_out) * w_fc` 곱 전체가 unsigned 로 계산되어
//            attn_out 이 256 의 배수가 아니면 결과가 틀렸다. tb_fc_layer 의 1.0, 2.0 입력은 우연히 통과했다.)
//   [구조]   x 입력 추가(잔차), 제곱 단계, 바이어스(fc_extra.mem 의 첫 워드) 추가.
//   [타이밍] 곱셈 두 번을 한 클럭에 직렬로 하지 않도록 파이프라인 1단을 넣었다. CALC 는 17클럭(예전 16클럭).
//
// 메모리 파일: fc_weights.mem (16워드), fc_extra.mem (1워드: 바이어스 b). Vivado 에서는 둘 다 설계 소스로 추가한다.
// ============================================================================
module fc_layer (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] attn_out [0:3][0:3],
    input  wire [15:0] x        [0:3][0:3],   // qkv_transform 에 들어간 것과 같은 window (top.sv 에서 연결)
    output reg  [15:0] score_abnormal,
    output reg         done
);

    typedef enum logic [1:0] {
        IDLE = 2'b00,
        CALC = 2'b01,
        DONE = 2'b10
    } state_t;

    state_t state, next_state;

    logic signed [15:0] w_fc     [0:15];     // [수정 1] signed
    logic signed [15:0] fc_extra [0:0];      // fc_extra[0] = 바이어스 b (Q8.8)

    initial begin
        //*************** vivado 구울 때 경로 수정 ******************//
        $readmemh("../mem/fc_weights.mem", w_fc);   // modelsim 경로
        $readmemh("../mem/fc_extra.mem",   fc_extra);
        // $readmemh("fc_weights.mem", w_fc);   // vivado 경로
        // $readmemh("fc_extra.mem",   fc_extra);
    end

    logic [4:0]         idx;                 // 0~16 (17클럭)
    logic signed [35:0] mac_val;             // 누산기
    logic signed [15:0] sq_reg;              // 1단 파이프라인: 직전 클럭에 만든 제곱값

    // 1단: 이번 원소의 (O + X)^2 [23:8]
    //   idx[3:2] = 행(센서), idx[1:0] = 열(시간). idx 가 16 일 때는 값이 쓰이지 않는다.
    wire        [15:0] f_sum   = attn_out[idx[3:2]][idx[1:0]] + x[idx[3:2]][idx[1:0]];
    wire signed [31:0] sq_full = $signed(f_sum) * $signed(f_sum);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= IDLE;
            idx            <= 5'd0;
            mac_val        <= 36'sd0;
            sq_reg         <= 16'sd0;
            score_abnormal <= 16'd0;
            done           <= 1'b0;
        end else begin
            state <= next_state;

            case (state)
                IDLE: begin
                    done    <= 1'b0;
                    idx     <= 5'd0;
                    mac_val <= 36'sd0;
                end
                CALC: begin
                    // 2단: 직전 클럭의 제곱값 x 가중치 누적 (idx==0 일 때는 아직 제곱값이 없다)
                    if (idx != 5'd0)
                        mac_val <= mac_val + ($signed(sq_reg) * w_fc[idx - 5'd1]);
                    // 1단 결과를 레지스터에 저장
                    sq_reg <= sq_full[23:8];
                    idx    <= idx + 5'd1;
                end
                DONE: begin
                    // Q8.8 복원([23:8]) 후 바이어스 가산 (16비트 wrap)
                    score_abnormal <= mac_val[23:8] + fc_extra[0];
                    done           <= 1'b1;
                end
                default: ;
            endcase
        end
    end

    // 다음 상태 결정 로직
    always_comb begin
        next_state = state;
        case (state)
            IDLE: if (start) next_state = CALC;
            CALC: if (idx == 5'd16) next_state = DONE;   // 16개 원소 + 파이프라인 1클럭
            DONE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

endmodule
