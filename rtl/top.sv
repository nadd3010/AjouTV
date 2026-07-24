module top (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,
    output wire       tx,
    output wire [3:0] led,
    output wire       buzzer
);

    // ============================================
    // 내부 신호 선언
    // ============================================

    // sensor_interface
    wire [7:0]  uart_data_out;
    wire        uart_data_valid;
    wire [15:0] scaled_out;
    wire        scaled_valid;

    // sliding_window
    wire [15:0] window [0:3][0:3];
    wire        window_ready;

    // qkv_transform
    wire [15:0] q [0:3][0:3];
    wire [15:0] k [0:3][0:3];
    wire [15:0] v [0:3][0:3];
    wire        qkv_done;

    // score_compute
    wire [15:0] score [0:3][0:3];
    wire        score_done;

    // softmax_lut
    wire [15:0] attn_weight [0:3][0:3];
    wire        softmax_done;

    // attn_output
    wire [15:0] attn_out [0:3][0:3];
    wire        attn_done;

    // fc_layer
    wire [15:0] score_abnormal;
    wire        classify_done;

    // argmax
    wire        label;

    // cause_identifier
    wire [17:0] col_sum [0:3];
    wire [1:0]  cause_index;

    // uart_tx
    wire        tx_done;

    // main_sequencer
    wire        attn_start;
    wire        classify_start;
    wire        tx_start;
    wire [2:0]  fsm_state;

    // ============================================
    // attention_engine 내부 체이닝 신호
    // qkv_done → score_compute start
    // score_done → softmax_lut start
    // softmax_done → attn_output start
    // ============================================

    // ============================================
    // 모듈 인스턴스
    // ============================================

    // --- sensor_interface ---
    uart_rx u_uart_rx (
        .clk(clk),
        .rst_n(rst_n),
        .rx(rx),
        .data_out(uart_data_out),
        .data_valid(uart_data_valid)
    );

    input_scaler u_input_scaler (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(uart_data_out),
        .data_valid(uart_data_valid),
        .scaled_out(scaled_out),
        .scaled_valid(scaled_valid)
    );

    // --- sliding_window ---
    sliding_window u_sliding_window (
        .clk(clk),
        .rst_n(rst_n),
        .sensor_data(scaled_out),
        .data_valid(scaled_valid),
        .window(window),
        .window_ready(window_ready)
    );

    // --- attention_engine ---
    qkv_transform u_qkv_transform (
        .clk(clk),
        .rst_n(rst_n),
        .start(attn_start),
        .x(window),
        .q(q),
        .k(k),
        .v(v),
        .done(qkv_done)
    );

    score_compute u_score_compute (
        .clk(clk),
        .rst_n(rst_n),
        .start(qkv_done),
        .q(q),
        .k(k),
        .score(score),
        .done(score_done)
    );

    softmax_lut u_softmax_lut (
        .clk(clk),
        .rst_n(rst_n),
        .start(score_done),
        .score(score),
        .attn_weight(attn_weight),
        .done(softmax_done)
    );

    attn_output u_attn_output (
        .clk(clk),
        .rst_n(rst_n),
        .start(softmax_done),
        .attn_weight(attn_weight),
        .v(v),
        .attn_out(attn_out),
        .done(attn_done)
    );

    // --- classifier ---
    fc_layer u_fc_layer (
        .clk(clk),
        .rst_n(rst_n),
        .start(classify_start),
        .attn_out(attn_out),
        .score_abnormal(score_abnormal),
        .done(classify_done)
    );

    argmax u_argmax (
        .score_abnormal(score_abnormal),
        .label(label)
    );

    cause_identifier u_cause_identifier (
        .clk(clk),
        .rst_n(rst_n),
        .attn_weight(attn_weight),
        .col_sum(col_sum),
        .cause_index(cause_index)
    );

    // --- output_interface ---
    led_controller u_led_controller (
        .label(label),
        .cause_index(cause_index),
        .led(led)
    );

    buzzer_driver u_buzzer_driver (
        .clk(clk),
        .rst_n(rst_n),
        .label(label),
        .buzzer(buzzer)
    );

    uart_tx u_uart_tx (
        .clk(clk),
        .rst_n(rst_n),
        .start(tx_start),
        .label(label),
        .col_sum(col_sum),
        .tx(tx),
        .done(tx_done)
    );

    // --- control_fsm ---
    main_sequencer u_main_sequencer (
        .clk(clk),
        .rst_n(rst_n),
        .window_ready(window_ready),
        .attn_done(attn_done),
        .classify_done(classify_done),
        .tx_done(tx_done),
        .attn_start(attn_start),
        .classify_start(classify_start),
        .tx_start(tx_start),
        .state(fsm_state)
    );

endmodule
