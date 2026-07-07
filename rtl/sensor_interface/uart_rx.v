module uart_rx (
    input  wire        clk,        // 50MHz
    input  wire        rst_n,      // active-low 리셋
    input  wire        rx,         // UART 수신 라인
    output reg  [7:0]  data_out,   // 수신된 8비트 데이터
    output reg         data_valid  // 1클럭 동안 HIGH
);

    // 50MHz / 115200 = 434 클럭/비트
    localparam CLKS_PER_BIT = 434;
    localparam HALF_BIT     = 217;

    // FSM 상태 정의
    localparam IDLE          = 3'd0;
    localparam RX_START_BIT  = 3'd1;
    localparam RX_DATA_BITS  = 3'd2;
    localparam RX_STOP_BIT   = 3'd3;
    localparam DONE          = 3'd4;

    reg [2:0]  state;
    reg [8:0]  clk_count;    // 최대 434까지 카운트
    reg [2:0]  bit_index;    // 0~7 데이터 비트
    reg [7:0]  rx_shift;     // 수신 중인 데이터

    // 메타스태빌리티 방지 2단 플립플롭
    reg rx_sync1, rx_sync2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_sync1 <= 1'b1;
            rx_sync2 <= 1'b1;
        end else begin
            rx_sync1 <= rx;
            rx_sync2 <= rx_sync1;
        end
    end

    // FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= IDLE;
            clk_count  <= 9'd0;
            bit_index  <= 3'd0;
            rx_shift   <= 8'd0;
            data_out   <= 8'd0;
            data_valid <= 1'b0;
        end else begin
            data_valid <= 1'b0;  // 기본값: LOW

            case (state)
                IDLE: begin
                    clk_count <= 9'd0;
                    bit_index <= 3'd0;
                    if (rx_sync2 == 1'b0)        // Start Bit 감지
                        state <= RX_START_BIT;
                end

                RX_START_BIT: begin
                    if (clk_count == HALF_BIT) begin  // 비트 정중앙
                        clk_count <= 9'd0;
                        if (rx_sync2 == 1'b0)         // 여전히 LOW → 유효
                            state <= RX_DATA_BITS;
                        else                           // 노이즈 → 복귀
                            state <= IDLE;
                    end else begin
                        clk_count <= clk_count + 9'd1;
                    end
                end

                RX_DATA_BITS: begin
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 9'd0;
                        rx_shift[bit_index] <= rx_sync2;  // LSB first
                        if (bit_index == 3'd7)
                            state <= RX_STOP_BIT;
                        else
                            bit_index <= bit_index + 3'd1;
                    end else begin
                        clk_count <= clk_count + 9'd1;
                    end
                end

                RX_STOP_BIT: begin
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 9'd0;
                        state     <= DONE;
                    end else begin
                        clk_count <= clk_count + 9'd1;
                    end
                end

                DONE: begin
                    data_out   <= rx_shift;
                    data_valid <= 1'b1;    // 1클럭 HIGH
                    state      <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
