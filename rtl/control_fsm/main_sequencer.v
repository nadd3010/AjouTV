module main_sequencer (
    input  wire clk,
    input  wire rst_n,
    input  wire window_ready,
    input  wire attn_done,
    input  wire classify_done,
    input  wire tx_done,
    output reg  attn_start,
    output reg  classify_start,
    output reg  tx_start,
    output reg  [2:0] state
);

// TODO: IDLERECEIVEWINDOWATTENTIONCLASSIFYOUTPUT 6상태 Moore FSM

endmodule
