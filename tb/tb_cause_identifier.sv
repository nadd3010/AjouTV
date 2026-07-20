`timescale 1ns / 1ps

module tb_cause_identifier();

    // 입력 및 출력 신호 선언
    logic        clk;
    logic        rst_n;
    logic [15:0] attn_weight [0:3][0:3];
    
    logic [17:0] col_sum [0:3];
    logic [1:0]  cause_index;

    // 테스트할 모듈(DUT) 인스턴스화
    cause_identifier dut (
        .clk(clk),
        .rst_n(rst_n),
        .attn_weight(attn_weight),
        .col_sum(col_sum),
        .cause_index(cause_index)
    );

    // 1. 클럭 생성 (조합 회로지만 인터페이스 통일성을 위해 제공)
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns 주기 클럭
    end

    // 2. 테스트 시나리오 인가
    initial begin
        // 초기화
        rst_n = 0;
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                attn_weight[i][j] = 16'd0;
            end
        end
        #10;
        
        rst_n = 1;
        #10;

        // --------------------------------------------------------------------
        // Test Case 1: 특정 센서(Index 1)의 가중치가 가장 높은 일반적인 상황
        // --------------------------------------------------------------------
        $display("==================================================");
        $display("[Test Case 1] Sensor 1 (Index 1) is Max");
        for (int i = 0; i < 4; i++) begin
            attn_weight[i][0] = 16'd10;
            attn_weight[i][1] = 16'd1000; // Sensor 1에 높은 가중치 부여
            attn_weight[i][2] = 16'd20;
            attn_weight[i][3] = 16'd5;
        end
        #10;
        $display("col_sum[0]: %0d, col_sum[1]: %0d, col_sum[2]: %0d, col_sum[3]: %0d", col_sum[0], col_sum[1], col_sum[2], col_sum[3]);
        $display("Expected cause_index: 1, Actual cause_index: %0d", cause_index);
        $display("==================================================\n");

        // --------------------------------------------------------------------
        // Test Case 2: 오버플로우 방지 확인 및 Sensor 3 (Index 3) 최대
        // --------------------------------------------------------------------
        $display("==================================================");
        $display("[Test Case 2] Overflow Check & Sensor 3 is Max");
        for (int i = 0; i < 4; i++) begin
            attn_weight[i][0] = 16'hFFFF; // 65535
            attn_weight[i][1] = 16'h1000;
            attn_weight[i][2] = 16'h2000;
            attn_weight[i][3] = 16'hFFFF; // 65535
        end
        // Sensor 0의 값을 조금 낮춰서 Sensor 3이 단독 최댓값이 되도록 설정
        attn_weight[3][0] = 16'hFFFE; 
        #10;
        // 16'hFFFF (65535) * 4 = 262140 (18비트 공간 필요)
        $display("col_sum[0]: %0d, col_sum[1]: %0d, col_sum[2]: %0d, col_sum[3]: %0d", col_sum[0], col_sum[1], col_sum[2], col_sum[3]);
        $display("Expected cause_index: 3, Actual cause_index: %0d", cause_index);
        $display("==================================================\n");

        // --------------------------------------------------------------------
        // Test Case 3: Sensor 2 (Index 2) 최대 및 랜덤 값 검증
        // --------------------------------------------------------------------
        $display("==================================================");
        $display("[Test Case 3] Sensor 2 (Index 2) is Max");
        for (int i = 0; i < 4; i++) begin
            attn_weight[i][0] = 16'd300;
            attn_weight[i][1] = 16'd800;
            attn_weight[i][2] = 16'd2000; // Sensor 2에 높은 가중치 부여
            attn_weight[i][3] = 16'd150;
        end
        #10;
        $display("col_sum[0]: %0d, col_sum[1]: %0d, col_sum[2]: %0d, col_sum[3]: %0d", col_sum[0], col_sum[1], col_sum[2], col_sum[3]);
        $display("Expected cause_index: 2, Actual cause_index: %0d", cause_index);
        $display("==================================================\n");

        // 시뮬레이션 종료
        #20;
        $stop;
    end

endmodule