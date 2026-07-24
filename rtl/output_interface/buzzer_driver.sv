module buzzer_driver (
    input  wire clk,
    input  wire rst_n,
    input  wire label,
    output reg  buzzer
);

    // 50MHz 기준 0.5초 = 25,000,000 클럭
    localparam BUZZER_DURATION = 25_000_000;

    reg [24:0] buzz_cnt;
    reg        buzzing;
    reg        label_prev;  // 이전 label 값 (엣지 감지용)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buzzer     <= 1'b0;
            buzz_cnt   <= 25'd0;
            buzzing    <= 1'b0;
            label_prev <= 1'b0;
        end else begin
            label_prev <= label;

            // label 0→1 상승 엣지 감지 → 부저 시작
            if (label && !label_prev) begin
                buzzing  <= 1'b1;
                buzz_cnt <= BUZZER_DURATION;
                buzzer   <= 1'b1;
            end else if (buzzing) begin
                if (buzz_cnt == 25'd0) begin
                    buzzing <= 1'b0;
                    buzzer  <= 1'b0;
                end else begin
                    buzz_cnt <= buzz_cnt - 25'd1;
                end
            end
        end
    end

endmodule
