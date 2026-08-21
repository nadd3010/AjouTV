module uart_tx (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire        label,
    input  wire [1:0]  cause_index,
    input  wire [17:0] col_sum [0:3],
    output reg         tx,
    output reg         done
);

    // 50MHz / 115200 baud
    localparam CLKS_PER_BIT = 434;

    // ===== 내부 바이트 전송 FSM =====
    localparam BIT_IDLE  = 2'd0;
    localparam BIT_START = 2'd1;
    localparam BIT_DATA  = 2'd2;
    localparam BIT_STOP  = 2'd3;

    reg [1:0]  bit_state;
    reg [8:0]  clk_cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  shift_reg;
    reg        byte_done;
    reg        byte_start;

    // 바이트 단위 전송기
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_state <= BIT_IDLE;
            tx        <= 1'b1;
            clk_cnt   <= 9'd0;
            bit_idx   <= 3'd0;
            byte_done <= 1'b0;
        end else begin
            byte_done <= 1'b0;

            case (bit_state)
                BIT_IDLE: begin
                    tx <= 1'b1;
                    if (byte_start) begin
                        bit_state <= BIT_START;
                        clk_cnt   <= 9'd0;
                    end
                end

                BIT_START: begin
                    tx <= 1'b0;
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 9'd0;
                        bit_idx <= 3'd0;
                        bit_state <= BIT_DATA;
                    end else
                        clk_cnt <= clk_cnt + 9'd1;
                end

                BIT_DATA: begin
                    tx <= shift_reg[bit_idx];
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 9'd0;
                        if (bit_idx == 3'd7)
                            bit_state <= BIT_STOP;
                        else
                            bit_idx <= bit_idx + 3'd1;
                    end else
                        clk_cnt <= clk_cnt + 9'd1;
                end

                BIT_STOP: begin
                    tx <= 1'b1;
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt   <= 9'd0;
                        byte_done <= 1'b1;
                        bit_state <= BIT_IDLE;
                    end else
                        clk_cnt <= clk_cnt + 9'd1;
                end
            endcase
        end
    end

    // ===== 9바이트 프로토콜 FSM =====
    // 전송 순서: label(1B) → col_sum[0] H/L → col_sum[1] H/L → col_sum[2] H/L → col_sum[3] H/L
    localparam PKT_IDLE = 4'd0;
    localparam PKT_SEND = 4'd1;
    localparam PKT_WAIT = 4'd2;
    localparam PKT_DONE = 4'd3;

    reg [3:0]  pkt_state;
    reg [3:0]  byte_idx;      // 0~8 (9바이트)
    reg [7:0]  tx_buf [0:8];  // 9바이트 버퍼

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pkt_state  <= PKT_IDLE;
            byte_idx   <= 4'd0;
            byte_start <= 1'b0;
            shift_reg  <= 8'd0;
            done       <= 1'b0;

            for (int i = 0; i < 9; i = i + 1) begin
            tx_buf[i] <= 8'd0;
            end
        end else begin
            done       <= 1'b0;
            byte_start <= 1'b0;

            case (pkt_state)
                PKT_IDLE: begin
                    if (start) begin
                        // 9바이트 버퍼 구성
                        tx_buf[0] <= {5'd0, cause_index, label};                   // label
                        tx_buf[1] <= col_sum[0][15:8];                 // col_sum[0] 상위
                        tx_buf[2] <= col_sum[0][7:0];                  // col_sum[0] 하위
                        tx_buf[3] <= col_sum[1][15:8];                 // col_sum[1] 상위
                        tx_buf[4] <= col_sum[1][7:0];                  // col_sum[1] 하위
                        tx_buf[5] <= col_sum[2][15:8];                 // col_sum[2] 상위
                        tx_buf[6] <= col_sum[2][7:0];                  // col_sum[2] 하위
                        tx_buf[7] <= col_sum[3][15:8];                 // col_sum[3] 상위
                        tx_buf[8] <= col_sum[3][7:0];                  // col_sum[3] 하위
                        byte_idx  <= 4'd0;
                        pkt_state <= PKT_SEND;
                    end
                end

                PKT_SEND: begin
                    shift_reg  <= tx_buf[byte_idx];
                    byte_start <= 1'b1;
                    pkt_state  <= PKT_WAIT;
                end

                PKT_WAIT: begin
                    if (byte_done) begin
                        if (byte_idx == 4'd8) begin
                            pkt_state <= PKT_DONE;
                        end else begin
                            byte_idx  <= byte_idx + 4'd1;
                            pkt_state <= PKT_SEND;
                        end
                    end
                end

                PKT_DONE: begin
                    done      <= 1'b1;
                    pkt_state <= PKT_IDLE;
                end
            endcase
        end
    end

endmodule
