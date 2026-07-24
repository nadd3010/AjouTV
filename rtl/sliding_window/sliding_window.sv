module sliding_window (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] sensor_data,    // Q8.8 from input_scaler
    input  wire        data_valid,
    output reg  [15:0] window [0:3][0:3],
    output reg         window_ready
);

    // 카운터: 행(row), 열(col)
    reg [1:0] row_cnt;
    reg [1:0] col_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_cnt      <= 2'd0;
            col_cnt      <= 2'd0;
            window_ready <= 1'b0;
        end else begin
            window_ready <= 1'b0;  // default: 1클럭 펄스

            if (data_valid) begin
                // 현재 위치에 데이터 저장
                window[row_cnt][col_cnt] <= sensor_data;

                if (col_cnt == 2'd3) begin
                    col_cnt <= 2'd0;

                    if (row_cnt == 2'd3) begin
                        // 4×4 전부 채움 → ready 신호
                        row_cnt      <= 2'd0;
                        window_ready <= 1'b1;
                    end else begin
                        row_cnt <= row_cnt + 2'd1;
                    end
                end else begin
                    col_cnt <= col_cnt + 2'd1;
                end
            end
        end
    end

endmodule
