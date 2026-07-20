`timescale 1ns / 1ps

module tb_attn_output;

    // 1. 입출력 신호 선언
    // [핵심] vsim-3906 에러 방지를 위해 다차원 배열에 signed 키워드를 의도적으로 제거했습니다.
    logic        clk;
    logic        rst_n;
    logic        start;
    logic [15:0] attn_weight [0:3][0:3];
    logic [15:0] v [0:3][0:3];
    wire  [15:0] attn_out [0:3][0:3];
    wire         done;

    // 2. DUT 연결
    attn_output dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .attn_weight(attn_weight),
        .v(v),
        .attn_out(attn_out),
        .done(done)
    );

    // 3. 클럭 생성 (100MHz)
    always #5 clk = ~clk;

    // 4. 테스트 시나리오
    initial begin
        // ==========================================
        // 초기화 (X값 전파 원천 차단)
        // ==========================================
        clk = 0;
        rst_n = 0;
        start = 0;
        
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                attn_weight[i][j] = 16'h0000;
                v[i][j]         = 16'h0000;
            end
        end

        // 리셋 해제
        #20 rst_n = 1;
        #10;

        // ==========================================
        // 테스트 데이터 주입 (Q8.8 포맷)
        // 직관적인 검증을 위해 attn_weight를 단위 행렬처럼 세팅
        // ==========================================
        // attn_weight[0] 행의 첫 번째 요소만 1.0으로 설정
        attn_weight[0][0] = 16'h0100; // 1.0 (Q8.8)
        
        // V 행렬의 0번 행에 데이터 세팅
        v[0][0] = 16'h0200; // 2.0 (Q8.8)
        v[0][1] = 16'h0300; // 3.0 (Q8.8)
        
        // 예상 결과: attn_out[0][0] = 0200, attn_out[0][1] = 0300, 나머지는 모두 0000

        // 연산 시작 신호 인가
        start = 1;
        #10 start = 0;

        // ==========================================
        // 무한 루프 방지 및 안전 종료 로직
        // ==========================================
        fork
            begin
                wait(done);
                #20;
                $display("========================================");
                $display("✅ 연산 완료! 파형에서 결과를 확인하세요.");
                $display("========================================");
            end
            begin
                #1000; // 1000ns 타임아웃
                $display("========================================");
                $display("🚨 [경고] 타임아웃! done 신호가 오지 않습니다.");
                $display("========================================");
            end
        join_any

        // 시뮬레이터 완전 종료 방지 ($finish 대신 $stop 사용)
        $stop;
    end

endmodule