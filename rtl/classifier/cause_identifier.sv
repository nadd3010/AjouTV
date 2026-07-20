module cause_identifier (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] attn_weight [0:3][0:3],
    output reg  [17:0] col_sum [0:3],
    output reg  [1:0]  cause_index
);

    // 내부 연산을 위한 logic 선언
    logic [17:0] temp_col_sum [0:3];
    logic [1:0]  max_idx;
    logic [17:0] max_val;
    integer i, j;

   // 1. 센서별 Attention Score 합산 (열 합산)
    always_comb begin
        // for문 내부에 int j 선언 (지역 변수화)
        for (int j = 0; j < 4; j = j + 1) begin
            temp_col_sum[j] = {2'b00, attn_weight[0][j]} + 
                              {2'b00, attn_weight[1][j]} + 
                              {2'b00, attn_weight[2][j]} + 
                              {2'b00, attn_weight[3][j]};
        end
    end

    // 2. 최대 기여 센서 식별 (비교기 트리)
    always_comb begin
        max_idx = 2'b00;
        max_val = temp_col_sum[0];

        // for문 내부에 int i 선언 (지역 변수화)
        for (int i = 1; i < 4; i = i + 1) begin
            if (temp_col_sum[i] > max_val) begin
                max_val = temp_col_sum[i];
                max_idx = i[1:0];
            end
        end
    end

    // 3. 출력 할당
    always_comb begin
        // 위 블록의 j와 완전히 독립적인 새로운 지역 변수 j
        for (int j = 0; j < 4; j = j + 1) begin
            col_sum[j] = temp_col_sum[j];
        end
        cause_index = max_idx;
    end
endmodule