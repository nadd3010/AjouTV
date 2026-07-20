`timescale 1ns / 1ps

module tb_argmax();

    // 입력 및 출력 포트 선언
    reg  [15:0] score_abnormal;
    wire        label;

    // 테스트할 모듈(UUT: Unit Under Test) 인스턴스화
    argmax uut (
        .score_abnormal(score_abnormal),
        .label(label)
    );

    // 시뮬레이션 블록
    initial begin
        $display("==================================================");
        $display("   Argmax (Comparator) Module Testbench Started   ");
        $display("   - Q8.8 Format, 임시 Threshold = 16'h0080 (0.5) ");
        $display("==================================================");

        // 초기화
        score_abnormal = 16'h0000;
        #10; 

        // --------------------------------------------------
        // Test Case 1: 확실한 정상 데이터 (Score < Threshold)
        // --------------------------------------------------
        score_abnormal = 16'h0000; // 0.0
        #10;
        $display("[TC1] Score: %04x ( 0.00) | Label: %b | Expected: 0", score_abnormal, label);

        score_abnormal = 16'h0040; // 0.25
        #10;
        $display("[TC2] Score: %04x ( 0.25) | Label: %b | Expected: 0", score_abnormal, label);

        // --------------------------------------------------
        // Test Case 2: 경계값 테스트 (Score == Threshold)
        // --------------------------------------------------
        score_abnormal = 16'h0080; // 0.5 (Threshold와 동일)
        // 설계상 (score > threshold)일 때만 1이므로, 같으면 0이 나와야 함
        #10;
        $display("[TC3] Score: %04x ( 0.50) | Label: %b | Expected: 0", score_abnormal, label);

        // --------------------------------------------------
        // Test Case 3: 확실한 이상 데이터 (Score > Threshold)
        // --------------------------------------------------
        score_abnormal = 16'h0081; // 0.5보다 아주 조금 큼
        #10;
        $display("[TC4] Score: %04x (~0.50+) | Label: %b | Expected: 1", score_abnormal, label);

        score_abnormal = 16'h0100; // 1.0
        #10;
        $display("[TC5] Score: %04x ( 1.00) | Label: %b | Expected: 1", score_abnormal, label);

        // --------------------------------------------------
        // Test Case 4: 음수 테스트 (부호 있는 연산 검증)
        // --------------------------------------------------
        // 신경망 결과값이 음수일 경우 16진수로는 앞자리가 F로 시작하여 매우 큰 양수처럼 보임
        // $signed 처리가 안 되어 있다면 label 1을 출력하는 오류가 발생함
        score_abnormal = 16'hFF00; // -1.0 (Q8.8 포맷)
        #10;
        $display("[TC6] Score: %04x (-1.00) | Label: %b | Expected: 0", score_abnormal, label);

        $display("==================================================");
        $display("             Testbench Finished                   ");
        $display("==================================================");
        
        $finish; // 시뮬레이션 종료
    end

endmodule