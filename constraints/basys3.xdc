###############################################################################
# AjouTV - Basys3 Constraints
# Target Board : Digilent Basys3
# FPGA         : Artix-7 XC7A35T
# System Clock : 100 MHz
###############################################################################

# 1. System Clock - 100 MHz
# Basys3 onboard oscillator: 100 MHz
set_property PACKAGE_PIN W5 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]

# 100 MHz = 10 ns period
create_clock -add -name sys_clk -period 10.000 -waveform {0 5} [get_ports clk]


# 2. Reset
# rst_n is ACTIVE LOW.

# 여기서는 SW0를 reset으로 사용:
#   SW0 = 0 -> Reset
#   SW0 = 1 -> Normal operation

# Basys3 SW0 = V17
set_property PACKAGE_PIN V17 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]


# 3. USB UART
# FPGA RX <- PC TX
# Basys3 USB-RS232 RX
set_property PACKAGE_PIN B18 [get_ports rx]
set_property IOSTANDARD LVCMOS33 [get_ports rx]

# FPGA TX -> PC RX
# Basys3 USB-RS232 TX
set_property PACKAGE_PIN A18 [get_ports tx]
set_property IOSTANDARD LVCMOS33 [get_ports tx]


# 4. LEDs
# LED0
set_property PACKAGE_PIN U16 [get_ports {led[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[0]}]

# LED1
set_property PACKAGE_PIN E19 [get_ports {led[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[1]}]

# LED2
set_property PACKAGE_PIN U19 [get_ports {led[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[2]}]

# LED3
set_property PACKAGE_PIN V19 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[3]}]


# 5. Buzzer
# Basys3에는 onboard buzzer가 없음.
# 아래 설정은 외부 buzzer 구동 신호를 Pmod JA1으로 출력하는 경우.
#
# JA1 = FPGA J1
set_property PACKAGE_PIN J1 [get_ports buzzer]
set_property IOSTANDARD LVCMOS33 [get_ports buzzer]