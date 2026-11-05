`timescale 1ns / 1ps

// ============================================================================
// tb_softmax_lut_vec : 골든 모델의 score(exp_score.mem)를 softmax_lut 에 직접 넣고 attn_weight(exp_aw.mem)와 비교.
//   상류 모듈의 영향 없이 softmax_lut 만 검증한다.  (make_rtl_vectors.py 가 만든 키트가 있는 폴더에서 실행)
// ============================================================================
module tb_softmax_lut_vec;

    localparam MAXW = 1024;
    reg clk = 0, rst_n = 0, start = 0;
    always #5 clk = ~clk;

    reg [15:0] nmem [0:0];
    reg [15:0] e_score [0:MAXW*16-1];
    reg [15:0] e_aw    [0:MAXW*16-1];
    reg signed [15:0] score [0:3][0:3];
    wire signed [15:0] attn_weight [0:3][0:3];
    wire done;
    integer n, w, i, j, bad, timeout, lat_sum;

    softmax_lut dut (.clk(clk), .rst_n(rst_n), .start(start), .score(score), .attn_weight(attn_weight), .done(done));

    initial begin
        $readmemh("../python/kit_fixed/vec_n.mem", nmem);
        n = nmem[0];
        $readmemh("../python/kit_fixed/exp_score.mem", e_score, 0, n*16-1);
        $readmemh("../python/kit_fixed/exp_aw.mem",    e_aw,    0, n*16-1);
        bad = 0; lat_sum = 0;
        for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) score[i][j] = 0;
        #20 rst_n = 1;
        #20;
        for (w = 0; w < n; w = w + 1) begin
            for (i = 0; i < 4; i = i + 1)
                for (j = 0; j < 4; j = j + 1)
                    score[i][j] = e_score[w*16 + i*4 + j];
            @(posedge clk); #1 start = 1;
            @(posedge clk); #1 start = 0;
            timeout = 0;
            while (!done && timeout < 200) begin @(posedge clk); timeout = timeout + 1; end
            lat_sum = lat_sum + timeout;
            #1;
            begin : cmp
                integer mism;
                mism = 0;
                for (i = 0; i < 4; i = i + 1)
                    for (j = 0; j < 4; j = j + 1)
                        if (attn_weight[i][j] !== e_aw[w*16 + i*4 + j]) begin
                            mism = mism + 1;
                            if (bad < 5) $display("  [MISMATCH] window %0d [%0d][%0d]: RTL=%h exp=%h", w, i, j,
                                                  attn_weight[i][j], e_aw[w*16 + i*4 + j]);
                        end
                if (mism != 0) bad = bad + 1;
            end
            repeat (3) @(posedge clk);
        end
        $display("softmax_lut unit test: %0d windows, %0d mismatches -> %0s  (avg %0d clocks)", n, bad, (bad == 0) ? "PASS" : "FAIL", lat_sum / n);
`ifdef __ICARUS__
        $finish;
`else
        $stop;
`endif
    end
endmodule