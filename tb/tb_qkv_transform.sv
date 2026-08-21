`timescale 1ns / 1ps

module tb_qkv_transform;

    // 1. 입출력 신호 선언
    // DUT의 포트 선언과 완벽히 일치하도록 signed 키워드를 제거하고 logic으로 통일했습니다.
    logic clk;
    logic rst_n;
    logic start;
    logic [15:0] x [0:3][0:3];
    logic [15:0] q [0:3][0:3], k [0:3][0:3], v [0:3][0:3];
    logic done;

    // 2. DUT (Device Under Test) 연결
    qkv_transform dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .x(x), .q(q), .k(k), .v(v), .done(done)
    );

    // 3. 클럭 생성 (100MHz)
    always #5 clk = ~clk;

    // 4. 테스트 시나리오
    integer i, j;
    initial begin
        // 초기화
        clk = 0; rst_n = 0; start = 0;
        
        // 입력 행렬 전체를 0으로 초기화하여 X 전파 방지
        for(i=0; i<4; i=i+1) begin
            for(j=0; j<4; j=j+1) begin
                x[i][j] = 16'h0000;
            end
        end

        #20 rst_n = 1; // 리셋 해제

        // 특정 데이터만 테스트용으로 세팅 (Q8.8 포맷)
        x[0][0] = 16'h0100; // 1.0 
        x[1][1] = 16'h0280; // 2.5 (예시)

        #10 start = 1; // 연산 시작!
        #10 start = 0;

        // done 신호가 1이 될 때까지 대기
        wait(done);
        
        #20 $display("계산 완료! 파형 창에서 Q, K, V 결과를 확인하세요.");
        $finish;
    end
    initial begin
        #30; // 리셋이 풀리고 메모리가 로드될 때까지 약간 대기
        $display("========================================");
        $display("=== 🕵️‍♂️ ModelSim이 읽은 weights_rom 내부 데이터 ===");
        for(int i=0; i<48; i++) begin
            $display("w_fc[%0d] = %h", i, dut.weights_rom[i]);
        end
        $display("========================================");
    end
endmodule