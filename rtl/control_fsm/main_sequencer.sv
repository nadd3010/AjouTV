module main_sequencer (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       window_ready,
    input  wire       attn_done,
    input  wire       classify_done,
    input  wire       tx_done,
    output reg        attn_start,
    output reg        classify_start,
    output reg        tx_start,
    output reg  [2:0] state
);

    // FSM 상태 정의
    localparam S_IDLE      = 3'd0;
    localparam S_ATTENTION = 3'd1;
    localparam S_CLASSIFY  = 3'd2;
    localparam S_OUTPUT    = 3'd3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            attn_start     <= 1'b0;
            classify_start <= 1'b0;
            tx_start       <= 1'b0;
        end else begin
            // 기본값: 1클럭 펄스
            attn_start     <= 1'b0;
            classify_start <= 1'b0;
            tx_start       <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (window_ready) begin
                        state      <= S_ATTENTION;
                        attn_start <= 1'b1;
                    end
                end

                S_ATTENTION: begin
                    if (attn_done) begin
                        state          <= S_CLASSIFY;
                        classify_start <= 1'b1;
                    end
                end

                S_CLASSIFY: begin
                    if (classify_done) begin
                        state    <= S_OUTPUT;
                        tx_start <= 1'b1;
                    end
                end

                S_OUTPUT: begin
                    if (tx_done) begin
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
