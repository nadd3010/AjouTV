module sliding_window (
    input  wire        clk,
    input  wire        rst_n,

    // sensor0 -> sensor1 -> sensor2 -> sensor3 순서로 들어온다고 가정
    input  wire [15:0] sensor_data,   // Q8.8 from input_scaler
    input  wire        data_valid,

    // window[sensor][time]
    // row 0 = sensor 0의 최근 4시점
    // row 1 = sensor 1의 최근 4시점
    // row 2 = sensor 2의 최근 4시점
    // row 3 = sensor 3의 최근 4시점
    output reg [15:0] window [0:3][0:3],

    output reg         window_ready
);

    // 현재 수신 중인 센서 번호
    reg [1:0] sensor_cnt;

    // 현재까지 완성된 time-step 수
    reg [2:0] time_filled;

    // 현재 시점의 4개 센서 값을 임시 저장
    reg [15:0] sensor_buf [0:3];

    integer i, j;

    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            sensor_cnt   <= 2'd0;
            time_filled  <= 3'd0;
            window_ready <= 1'b0;

            for (i = 0; i < 4; i = i + 1) begin
                sensor_buf[i] <= 16'd0;

                for (j = 0; j < 4; j = j + 1) begin
                    window[i][j] <= 16'd0;
                end
            end

        end else begin

            // 기본적으로 0, 새로운 window가 완성될 때만 1클럭 HIGH
            window_ready <= 1'b0;

            if (data_valid) begin

                // 현재 시점의 센서 값 저장
                sensor_buf[sensor_cnt] <= sensor_data;

                // sensor 3까지 받으면 한 시점의 데이터 완성
                if (sensor_cnt == 2'd3) begin

                    // -----------------------------------------
                    // Sensor × Time Sliding Window
                    //
                    // 각 sensor row 내부에서 time을 shift
                    //
                    // [t1 t2 t3 t4]
                    //      ↓
                    // [t2 t3 t4 t5]
                    // -----------------------------------------

                    for (i = 0; i < 4; i = i + 1) begin
                        for (j = 0; j < 3; j = j + 1) begin
                            window[i][j] <= window[i][j+1];
                        end
                    end

                    // -----------------------------------------
                    // 가장 최신 시점 값을 마지막 column에 삽입
                    // -----------------------------------------

                    window[0][3] <= sensor_buf[0];
                    window[1][3] <= sensor_buf[1];
                    window[2][3] <= sensor_buf[2];

                    // 마지막 센서는 non-blocking assignment 때문에
                    // sensor_buf[3]이 아직 이전 값이므로
                    // 현재 sensor_data를 직접 사용
                    window[3][3] <= sensor_data;

                    sensor_cnt <= 2'd0;

                    // 최초 4개 time-step이 채워질 때까지 count
                    if (time_filled < 3'd4)
                        time_filled <= time_filled + 3'd1;

                    // 최초 4번째 시점부터 매 새로운 시점마다
                    // 새로운 4×4 window 생성
                    if (time_filled >= 3'd3)
                        window_ready <= 1'b1;

                end else begin

                    sensor_cnt <= sensor_cnt + 2'd1;

                end
            end
        end
    end

endmodule