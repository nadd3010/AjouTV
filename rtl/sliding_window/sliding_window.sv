module sliding_window (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] sensor_data,
    input  wire        data_valid,
    output reg  [15:0] window [0:3][0:3],
    output reg         window_ready
);

    reg [1:0] row_cnt;
    reg [1:0] col_cnt;

    integer i, j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_cnt      <= 2'd0;
            col_cnt      <= 2'd0;
            window_ready <= 1'b0;
            // ★ window 배열을 리셋에서 명시적으로 0 초기화
            for (i = 0; i < 4; i = i + 1)
                for (j = 0; j < 4; j = j + 1)
                    window[i][j] <= 16'd0;
        end else begin
            window_ready <= 1'b0;

            if (data_valid) begin
                window[row_cnt][col_cnt] <= sensor_data;

                if (col_cnt == 2'd3) begin
                    col_cnt <= 2'd0;
                    if (row_cnt == 2'd3) begin
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
