module led_controller (
    input  wire        label,
    input  wire [1:0]  cause_index,
    output reg  [3:0]  led
);

    // label=1(이상) → cause_index에 해당하는 LED만 ON (one-hot)
    // label=0(정상) → 모든 LED OFF
    always @(*) begin
        if (label)
            led = 4'b0001 << cause_index;  // one-hot
        else
            led = 4'b0000;
    end

endmodule
