`timescale 1ns / 1ps

// ============================================================================
// tb_core_e2e : UART/슬라이딩 윈도우를 제외한 코어 데이터패스(top.sv 와 같은 배선)를 골든 모델 벡터와 비교한다.
//   main_sequencer -> qkv_transform -> score_compute -> softmax_lut -> attn_output -> fc_layer -> argmax / cause_identifier
//
// 사용법: 시뮬레이션 작업 폴더에 make_rtl_vectors.py 가 만든 키트(*.mem)를 두고 실행한다.
//   수정한 RTL(res_square) : +define+PATCHED   (fc_layer 에 x 포트를 연결)
//   현재 main RTL          : (define 없음)
// 단계별(q, k, v, score, attn_weight, attn_out, z, label, cause) 불일치 개수와 처음 몇 건을 출력한다.
// ============================================================================
module tb_core_e2e;

    localparam MAXW = 1024;

    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;

    // ---------------- 벡터 ----------------
    reg [15:0] nmem [0:0];
    reg [15:0] vx      [0:MAXW*16-1];
    reg [15:0] e_q     [0:MAXW*16-1];
    reg [15:0] e_k     [0:MAXW*16-1];
    reg [15:0] e_v     [0:MAXW*16-1];
    reg [15:0] e_score [0:MAXW*16-1];
    reg [15:0] e_aw    [0:MAXW*16-1];
    reg [15:0] e_ao    [0:MAXW*16-1];
    reg [15:0] e_z     [0:MAXW-1];
    reg [15:0] e_label [0:MAXW-1];
    reg [15:0] e_cause [0:MAXW-1];
    integer n;

    // ---------------- DUT 신호 (top.sv 와 같은 이름) ----------------
    reg  [15:0] x [0:3][0:3];
    wire [15:0] q [0:3][0:3];
    wire [15:0] k [0:3][0:3];
    wire [15:0] v [0:3][0:3];
    wire signed [15:0] score [0:3][0:3];
    wire signed [15:0] attn_weight [0:3][0:3];
    wire [15:0] attn_out [0:3][0:3];
    wire [15:0] score_abnormal;
    wire        label;
    wire [17:0] col_sum [0:3];
    wire [1:0]  cause_index;

    wire qkv_done, score_done, softmax_done, attn_done, classify_done;
    wire attn_start, classify_start, tx_start;
    wire [2:0] fsm_state;
    reg  window_ready = 0;
    reg  tx_done = 0;

    main_sequencer u_seq (
        .clk(clk), .rst_n(rst_n),
        .window_ready(window_ready), .attn_done(attn_done), .classify_done(classify_done), .tx_done(tx_done),
        .attn_start(attn_start), .classify_start(classify_start), .tx_start(tx_start), .state(fsm_state)
    );

    qkv_transform   u_qkv  (.clk(clk), .rst_n(rst_n), .start(attn_start), .x(x), .q(q), .k(k), .v(v), .done(qkv_done));
    score_compute   u_sc   (.clk(clk), .rst_n(rst_n), .start(qkv_done), .q(q), .k(k), .score(score), .done(score_done));
    softmax_lut     u_sm   (.clk(clk), .rst_n(rst_n), .start(score_done), .score(score), .attn_weight(attn_weight), .done(softmax_done));
    attn_output     u_ao   (.clk(clk), .rst_n(rst_n), .start(softmax_done), .attn_weight(attn_weight), .v(v), .attn_out(attn_out), .done(attn_done));
`ifdef PATCHED
    fc_layer        u_fc   (.clk(clk), .rst_n(rst_n), .start(classify_start), .attn_out(attn_out), .x(x),
                            .score_abnormal(score_abnormal), .done(classify_done));
`else
    fc_layer        u_fc   (.clk(clk), .rst_n(rst_n), .start(classify_start), .attn_out(attn_out), .x(x),
                            .score_abnormal(score_abnormal), .done(classify_done));
`endif
    argmax          u_am   (.score_abnormal(score_abnormal), .label(label));
    cause_identifier u_ci  (.clk(clk), .rst_n(rst_n), .attn_weight(attn_weight), .col_sum(col_sum), .cause_index(cause_index));

    // ---------------- 비교 ----------------
    integer bad [0:8];                 // 단계별 불일치 윈도우 수
    integer shown [0:8];
    reg [15:0] got [0:15];
    integer w, i, j, s, t, timeout;
    integer lat_min, lat_max, lat_sum;
    integer fd;
    reg [8*10-1:0] sname [0:8];

    task automatic compare16(input integer stage, input integer win);
        integer e, mism;
        reg [15:0] exp_v;
        begin
            mism = 0;
            for (e = 0; e < 16; e = e + 1) begin
                case (stage)
                    0: exp_v = e_q[win*16+e];
                    1: exp_v = e_k[win*16+e];
                    2: exp_v = e_v[win*16+e];
                    3: exp_v = e_score[win*16+e];
                    4: exp_v = e_aw[win*16+e];
                    default: exp_v = e_ao[win*16+e];
                endcase
                if (got[e] !== exp_v) begin
                    mism = mism + 1;
                    if (shown[stage] < 3) begin
                        $display("  [MISMATCH] %0s window %0d elem %0d: RTL=%h exp=%h", sname[stage], win, e, got[e], exp_v);
                        shown[stage] = shown[stage] + 1;
                    end
                end
            end
            if (mism != 0) bad[stage] = bad[stage] + 1;
        end
    endtask

    initial begin
        sname[0] = "q"; sname[1] = "k"; sname[2] = "v"; sname[3] = "score"; sname[4] = "attn_w";
        sname[5] = "attn_out"; sname[6] = "z"; sname[7] = "label"; sname[8] = "cause";
        for (s = 0; s < 9; s = s + 1) begin bad[s] = 0; shown[s] = 0; end

        $readmemh("../python/kit_fixed/vec_n.mem", nmem);
        n = nmem[0];
        if (n > MAXW) begin
            $display("[ERROR] window count %0d exceeds MAXW(%0d). Increase MAXW.", n, MAXW);
            $finish;
        end
        $readmemh("../python/kit_fixed/vec_x.mem",     vx,      0, n*16-1);
        $readmemh("../python/kit_fixed/exp_q.mem",     e_q,     0, n*16-1);
        $readmemh("../python/kit_fixed/exp_k.mem",     e_k,     0, n*16-1);
        $readmemh("../python/kit_fixed/exp_v.mem",     e_v,     0, n*16-1);
        $readmemh("../python/kit_fixed/exp_score.mem", e_score, 0, n*16-1);
        $readmemh("../python/kit_fixed/exp_aw.mem",    e_aw,    0, n*16-1);
        $readmemh("../python/kit_fixed/exp_ao.mem",    e_ao,    0, n*16-1);
        $readmemh("../python/kit_fixed/exp_z.mem",     e_z,     0, n-1);
        $readmemh("../python/kit_fixed/exp_label.mem", e_label, 0, n-1);
        $readmemh("../python/kit_fixed/exp_cause.mem", e_cause, 0, n-1);

        for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) x[i][j] = 16'd0;
        fd = $fopen("rtl_trace.txt", "w");
        lat_min = 1000000; lat_max = 0; lat_sum = 0;

        #50 rst_n = 1;
        #50;

        for (w = 0; w < n; w = w + 1) begin
            // 윈도우 입력 (sliding_window 가 만드는 것과 같은 4x4, 행 = 센서, 열 = 시간)
            for (i = 0; i < 4; i = i + 1)
                for (j = 0; j < 4; j = j + 1)
                    x[i][j] = vx[w*16 + i*4 + j];
            @(posedge clk); #1 window_ready = 1;
            @(posedge clk); #1 window_ready = 0;

            // classify_done 까지 대기 (타임아웃 3000클럭)
            timeout = 0;
            while (!classify_done && timeout < 3000) begin
                @(posedge clk); timeout = timeout + 1;
            end
            if (timeout >= 3000) begin
                $display("[ERROR] window %0d: classify_done never arrived (timeout). fsm_state=%0d", w, fsm_state);
                $finish;
            end

            if (timeout < lat_min) lat_min = timeout;
            if (timeout > lat_max) lat_max = timeout;
            lat_sum = lat_sum + timeout;

            // 단계별 비교 (classify_done 가 1인 시점에 모든 단계 결과가 유효)
            for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) got[i*4+j] = q[i][j];
            compare16(0, w);
            for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) got[i*4+j] = k[i][j];
            compare16(1, w);
            for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) got[i*4+j] = v[i][j];
            compare16(2, w);
            for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) got[i*4+j] = score[i][j];
            compare16(3, w);
            for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) got[i*4+j] = attn_weight[i][j];
            compare16(4, w);
            for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) got[i*4+j] = attn_out[i][j];
            compare16(5, w);
            if (score_abnormal !== e_z[w]) begin
                bad[6] = bad[6] + 1;
                if (shown[6] < 5) begin
                    $display("  [MISMATCH] z window %0d: RTL=%h exp=%h", w, score_abnormal, e_z[w]);
                    shown[6] = shown[6] + 1;
                end
            end
            if ({15'd0, label} !== e_label[w]) bad[7] = bad[7] + 1;
            if ({14'd0, cause_index} !== e_cause[w]) bad[8] = bad[8] + 1;
            $fdisplay(fd, "%0d z=%h label=%b cause=%0d", w, score_abnormal, label, cause_index);

            // main_sequencer 를 IDLE 로 되돌림 (UART 송신 완료 흉내)
            timeout = 0;
            while (!tx_start && timeout < 20) begin @(posedge clk); timeout = timeout + 1; end
            @(posedge clk); #1 tx_done = 1;
            @(posedge clk); #1 tx_done = 0;
            repeat (3) @(posedge clk);
        end

        $fclose(fd);
        $display("====================================================");
        $display(" Core datapath check: %0d windows", n);
        for (s = 0; s < 9; s = s + 1)
            $display("  %0s : mismatched windows %0d / %0d  %0s", sname[s], bad[s], n, (bad[s] == 0) ? "PASS" : "FAIL");
        $display("  latency (window_ready -> classify_done): min %0d / avg %0d / max %0d clocks (avg about %0d ns at 100MHz)",
                 lat_min, lat_sum / n, lat_max, (lat_sum / n) * 10);
        t = 0;
        for (s = 0; s < 9; s = s + 1) t = t + bad[s];
        if (t == 0) $display(" [RESULT] ALL PASS");
        else        $display(" [RESULT] FAIL");
        $display("====================================================");
`ifdef __ICARUS__
        $finish;
`else
        $stop;
`endif
    end

endmodule