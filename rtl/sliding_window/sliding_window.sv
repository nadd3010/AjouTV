module sliding_window (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] sensor_data,    // Q8.8 from input_scaler
    input  wire        data_valid,
    output reg  [15:0] window [0:3][0:3],
    output reg         window_ready
);

    reg [1:0] col_cnt;       // 0~3: 현재 행 내 센서 인덱스
    reg [2:0] row_filled;    // 채워진 행 수 (0~4 표현 위해 3비트)
    reg [15:0] row_buf [0:3]; // 현재 수신 중인 1행 임시 버퍼

    integer i, j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col_cnt      <= 2'd0;
            row_filled   <= 3'd0;
            window_ready <= 1'b0;
            for (i = 0; i < 4; i = i + 1) begin
                row_buf[i] <= 16'd0;
                for (j = 0; j < 4; j = j + 1)
                    window[i][j] <= 16'd0;
            end
        end else begin
            window_ready <= 1'b0;  // 기본값: 1클럭 펄스

            if (data_valid) begin
                // 현재 행 버퍼에 센서 데이터 저장
                row_buf[col_cnt] <= sensor_data;

                if (col_cnt == 2'd3) begin
                    // 1행 완성 → FIFO 시프트
                    // 기존 행들을 한 칸씩 위로 밀기 (행0이 가장 오래된 데이터)
                    // 행0 ← 행1 ← 행2 ← 행3 ← 새 행
                    for (i = 0; i < 3; i = i + 1)
                        for (j = 0; j < 4; j = j + 1)
                            window[i][j] <= window[i+1][j];

                    // 새 행을 맨 아래(행3)에 삽입
                    window[3][0] <= row_buf[0];
                    window[3][1] <= row_buf[1];
                    window[3][2] <= row_buf[2];
                    window[3][3] <= sensor_data;  // 마지막 센서는 방금 받은 값

                    col_cnt <= 2'd0;

                    // 채워진 행 수 추적
                    if (row_filled < 3'd4)
                        row_filled <= row_filled + 3'd1;

                    // 4행 이상 채워졌으면 매번 ready
                    if (row_filled >= 3'd3)  // 다음 클럭에서 4가 되므로 3에서 판정
                        window_ready <= 1'b1;

                end else begin
                    col_cnt <= col_cnt + 2'd1;
                end
            end
        end
    end

endmodule
