`timescale 1ns / 1ps

module tb_fc_layer;

    // 1. DUT 연결용 포트 선언
    logic        clk;
    logic        rst_n;
    logic        start;
    logic [15:0] attn_out [0:3][0:3];
    
    logic [15:0] score_abnormal;
    logic        done;

    // 2. DUT(Design Under Test) 인스턴스화
    fc_layer uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .attn_out(attn_out),
        .score_abnormal(score_abnormal),
        .done(done)
    );

    // 3. 클럭 생성 (100MHz 기준, 10ns 주기)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // 4. 테스트 시나리오 입력 및 자동 검증
    initial begin
        // 초기화
        rst_n = 0;
        start = 0;
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                attn_out[i][j] = 16'd0;
            end
        end

        // 리셋 해제
        #20;
        rst_n = 1;
        #20;

        // --------------------------------------------------------
        // Test Case 1: 모든 Attention Out 입력이 1.0 인 경우
        // 실제 가중치 총합은 6.0 (16'h0600) 입니다.
        // --------------------------------------------------------
        $display("========================================");
        $display("=== Test Case 1: All Inputs = 1.0 ===");
        $display("========================================");
        
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                attn_out[i][j] = 16'h0100; 
            end
        end
        
        start = 1;
        #10;
        start = 0; // 1클럭 start 펄스 인가

        // 연산이 끝날 때까지 대기
        wait(done == 1'b1);
        
        #10; // 결과 확인을 위한 여유 시간
        $display("Expected Score : 0600 (6.0 in Q8.8)");
        $display("Actual Score   : %h", score_abnormal);

        if (score_abnormal == 16'h0600)
            $display("-> [RESULT] PASS! \n");
        else
            $display("-> [RESULT] FAIL! \n");


        // --------------------------------------------------------
        // Test Case 2: 모든 Attention Out 입력이 2.0 인 경우
        // 6.0 * 2.0 = 12.0 이므로, 예상 결과는 16'h0C00 입니다.
        // --------------------------------------------------------
        #30; // 다음 테스트 전 여유 시간
        $display("========================================");
        $display("=== Test Case 2: All Inputs = 2.0 ===");
        $display("========================================");
        
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                attn_out[i][j] = 16'h0200; // 2.0
            end
        end
        
        start = 1;
        #10;
        start = 0;

        // 연산 대기
        wait(done == 1'b1);
        
        #10;
        $display("Expected Score : 0C00 (12.0 in Q8.8)");
        $display("Actual Score   : %h", score_abnormal);

        // 기댓값을 0900에서 0C00으로 수정
        if (score_abnormal == 16'h0C00)
            $display("-> [RESULT] PASS! \n");
        else
            $display("-> [RESULT] FAIL! \n");

        // 시뮬레이션 종료
        #50;
        $display("Simulation Finished Successfully.");
        $stop; 
    end

endmodule