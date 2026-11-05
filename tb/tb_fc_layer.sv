`timescale 1ns / 1ps

// ============================================================================
// tb_fc_layer (res_square 구조용으로 교체한 단위 테스트)
//   골든 모델이 만든 attn_out(exp_ao.mem)과 window(vec_x.mem)를 fc_layer 에 직접 넣고 z(exp_z.mem)와 비교한다.
//   상류 모듈의 영향 없이 fc_layer 만 검증한다.  (make_rtl_vectors.py 가 만든 키트가 있는 폴더에서 실행)
// ============================================================================
module tb_fc_layer;

    localparam MAXW = 1024;
    reg clk = 0, rst_n = 0, start = 0;
    always #5 clk = ~clk;

    reg [15:0] nmem [0:0];
    reg [15:0] vx  [0:MAXW*16-1];
    reg [15:0] e_ao [0:MAXW*16-1];
    reg [15:0] e_z [0:MAXW-1];
    reg [15:0] attn_out [0:3][0:3];
    reg [15:0] x        [0:3][0:3];
    wire [15:0] score_abnormal;
    wire done;
    integer n, w, i, j, bad, timeout;

    fc_layer uut (.clk(clk), .rst_n(rst_n), .start(start), .attn_out(attn_out), .x(x),
                  .score_abnormal(score_abnormal), .done(done));

    initial begin
        $readmemh("../python/kit_fixed/vec_n.mem", nmem);
        n = nmem[0];
        $readmemh("../python/kit_fixed/vec_x.mem",  vx,   0, n*16-1);
        $readmemh("../python/kit_fixed/exp_ao.mem", e_ao, 0, n*16-1);
        $readmemh("../python/kit_fixed/exp_z.mem",  e_z,  0, n-1);
        bad = 0;
        for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) begin attn_out[i][j] = 0; x[i][j] = 0; end
        #20 rst_n = 1;
        #20;
        for (w = 0; w < n; w = w + 1) begin
            for (i = 0; i < 4; i = i + 1)
                for (j = 0; j < 4; j = j + 1) begin
                    attn_out[i][j] = e_ao[w*16 + i*4 + j];
                    x[i][j]        = vx[w*16 + i*4 + j];
                end
            @(posedge clk); #1 start = 1;
            @(posedge clk); #1 start = 0;
            timeout = 0;
            while (!done && timeout < 100) begin @(posedge clk); timeout = timeout + 1; end
            #1;
            if (score_abnormal !== e_z[w]) begin
                bad = bad + 1;
                if (bad <= 5) $display("  [MISMATCH] window %0d: RTL=%h exp=%h", w, score_abnormal, e_z[w]);
            end
            repeat (3) @(posedge clk);
        end
        $display("fc_layer unit test: %0d windows, %0d mismatches -> %0s", n, bad, (bad == 0) ? "PASS" : "FAIL");
        // 부호 확인: 모든 attn_out = 0x0040(0.25), x = 0 일 때 예전 RTL 은 부호 오류로 z 의 최상위 비트가 켜졌다 (0x801C).
        $display("(note) fc_weights.mem[0] = %h, bias fc_extra.mem[0] = %h", uut.w_fc[0], uut.fc_extra[0]);
`ifdef __ICARUS__
        $finish;
`else
        $stop;
`endif
    end
endmodule