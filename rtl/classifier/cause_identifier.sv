
`timescale 1ns / 1ps

module cause_identifier (
    input  wire        clk,
    input  wire        rst_n,
    input  wire signed [15:0] attn_weight [0:3][0:3],
    output reg  [17:0] col_sum [0:3],
    output reg  [1:0]  cause_index
);

    // =====================================================
    // 1. Stage 1: Column Sum
    //
    // 각 센서별 Attention Weight 열 합산
    // 4개 항을 2개씩 묶어 병렬 덧셈
    // =====================================================

    // 16bit + 16bit = 17bit
    logic [16:0] sum01 [0:3];
    logic [16:0] sum23 [0:3];

    // 최종 열 합계: 최대 18bit
    logic [17:0] next_col_sum [0:3];

    always_comb begin
        for (int j = 0; j < 4; j++) begin

            // 첫 번째 부분합
            sum01[j] =
                {1'b0, attn_weight[0][j]} +
                {1'b0, attn_weight[1][j]};

            // 두 번째 부분합
            sum23[j] =
                {1'b0, attn_weight[2][j]} +
                {1'b0, attn_weight[3][j]};

            // 두 부분합 누적
            next_col_sum[j] =
                {1'b0, sum01[j]} +
                {1'b0, sum23[j]};

        end
    end

    // =====================================================
    // 2. Stage 2: Maximum Finder
    //
    // col_sum 레지스터를 이용해 최대값 비교
    //
    // 비교기 트리:
    //   col_sum[0] vs col_sum[1]
    //   col_sum[2] vs col_sum[3]
    //            ↓
    //      두 최대값 비교
    //
    // 동점이면 작은 인덱스 우선
    // =====================================================

    logic [17:0] max01_val;
    logic [17:0] max23_val;

    logic [1:0] max01_idx;
    logic [1:0] max23_idx;

    logic [1:0] next_cause_index;

    always_comb begin

        // -----------------------------------------
        // Compare 0 vs 1
        // -----------------------------------------
        if (col_sum[0] >= col_sum[1]) begin
            max01_val = col_sum[0];
            max01_idx = 2'd0;
        end else begin
            max01_val = col_sum[1];
            max01_idx = 2'd1;
        end

        // -----------------------------------------
        // Compare 2 vs 3
        // -----------------------------------------
        if (col_sum[2] >= col_sum[3]) begin
            max23_val = col_sum[2];
            max23_idx = 2'd2;
        end else begin
            max23_val = col_sum[3];
            max23_idx = 2'd3;
        end

        // -----------------------------------------
        // Final Maximum
        // -----------------------------------------
        if (max01_val >= max23_val) begin
            next_cause_index = max01_idx;
        end else begin
            next_cause_index = max23_idx;
        end

    end

    // =====================================================
    // 3. Pipeline Registers
    //
    // Stage 1: Column Sum 저장
    // Stage 2: Cause Index 저장
    // =====================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            cause_index <= 2'd0;

            for (int j = 0; j < 4; j++) begin
                col_sum[j] <= 18'd0;
            end

        end else begin

            // -----------------------------------------
            // Stage 1: 열 합산 결과 저장
            // -----------------------------------------
            for (int j = 0; j < 4; j++) begin
                col_sum[j] <= next_col_sum[j];
            end

            // -----------------------------------------
            // Stage 2: 최대 기여 센서 저장
            //
            // Nonblocking Assignment 특성상
            // 이전 클럭에서 저장된 col_sum을 사용
            // -----------------------------------------
            cause_index <= next_cause_index;

        end
    end

endmodule
