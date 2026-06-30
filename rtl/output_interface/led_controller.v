module led_controller (
    input  wire       label,
    input  wire [1:0] cause_index,
    output reg  [3:0] led
);

// TODO: 이상 시 cause_index에 해당하는 LED 점등 (one-hot)

endmodule
