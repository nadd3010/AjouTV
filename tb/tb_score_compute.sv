`timescale 1ns / 1ps

module tb_score_compute;

    // 1. 입출력 신호 선언 (포트 충돌 방지를 위해 logic 자료형 사용)
    logic        clk;
    logic        rst_n;
    logic        start;
    logic [15:0] q [0:3][0:3];
    logic [15:0] k [0:3][0:3];
    
    logic [15:0] score [0:3][0:3];
    logic        done;

    // 2. DUT (Device Under Test) 연결
    score_compute dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .q(q),
        .k(k),
        .score(score),
        .done(done)
    );

    // 3. 클럭 생성 (100MHz)
    always #5 clk = ~clk;

    // 4. 테스트 시나리오
    initial begin
        // [초기화]
        clk = 0;
        rst_n = 0;
        start = 0;
        
        // 이전 모듈에서 배운 핵심: 붉은색 X(Unknown) 파형 전염을 막기 위한 초기화
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 4; j++) begin
                q[i][j] = 16'h0000;
                k[i][j] = 16'h0000;
            end
        end

        // 리셋 해제
        #20 rst_n = 1;
        
        // [입력 데이터 세팅] (Q8.8 포맷)
        // 시뮬레이션 파형에서 눈으로 즉각적인 검증이 가능하도록 직관적인 값을 넣습니다.
        // 예: q[0][0] = 1.0, k[0][0] = 1.0
        // 내적하면 1.0 이지만, 스케일링(>>> 1)을 거치므로 최종 score[0][0]은 0.5(16'h0080)가 나와야 합니다.
        q[0][0] = 16'h0100; // 1.0
        q[0][1] = 16'h0200; // 2.0
        
        k[0][0] = 16'h0100; // 1.0 (K는 설계상 내부에서 전치 행렬로 참조됨)
        k[0][1] = 16'h0080; // 0.5

        // [연산 시작]
        #10 start = 1;
        #10 start = 0; // 1클럭 펄스 인가

        // [완료 대기] done 신호가 1이 될 때까지 대기
        wait(done);
        
        #20;
        $display("========================================");
        $display("score_compute 연산 완료! 파형을 확인하세요.");
        $display("========================================");
        $finish;
    end

endmodule