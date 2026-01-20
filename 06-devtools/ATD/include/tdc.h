#ifndef TDC_H_
#define TDC_H_

#include <cstdint>
#include "common.h"
#include "serial_dev.h"

// TDC
//  This unit implements an interface to a Time-to-Digital Converter (TDC), such
//  as the TI TDC7201.  The unit is still a work-in-progress.
//
// Copyright (C) 2021, Lawrence Livermore National Security, LLC.
//  All rights reserved. LLNL-CODE-837067
//
// The Department of Homeland Security sponsored the production of this
//  material under DOE Contract Number DE-AC52-07N427344 for the management
//  and operation of Lawrence Livermore National Laboratory. Contract no.
//  DE-AC52-07NA27344 is between the U.S. Department of Energy (DOE) and
//  Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.
//  See license for disclaimers, notice of U.S. Government Rights and license
//  terms and conditions.

namespace tmon
{

enum class TdcCal
{
    unless_interrupted, always
};

enum class TdcEdge
{
    rising, falling
};

enum class TdcMeasMode
{
    mode1, mode2
};

enum class TdcRegister : std::uint8_t
{
    TDCx_CONFIG1            = 0x00,
    TDCx_CONFIG2            = 0x01,
    TDCx_INT_STATUS         = 0x02,
    TDCx_INT_MASK           = 0x03,
    TDCx_COARSE_CNTR_OVF_H  = 0x04,
    TDCx_COARSE_CNTR_OVF_L  = 0x05,
    TDCx_CLOCK_CNTR_OVF_H   = 0x06,
    TDCx_CLOCK_CNTR_OVF_L   = 0x07,
    TDCx_CLOCK_CNTR_STOP_MASK_H = 0x08,
    TDCx_CLOCK_CNTR_STOP_MASK_L = 0x09,
    TDCx_TIME1                  = 0x10,
    TDCx_CLOCK_COUNT1           = 0x11,
    TDCx_TIME2                  = 0x12,
    TDCx_CLOCK_COUNT2           = 0x13,
    TDCx_TIME3                  = 0x14,
    TDCx_CLOCK_COUNT3           = 0x15,
    TDCx_TIME4                  = 0x16,
    TDCx_CLOCK_COUNT4           = 0x17,
    TDCx_TIME5                  = 0x18,
    TDCx_CLOCK_COUNT5           = 0x19,
    TDCx_TIME6                  = 0x1A,
    TDCx_CALIBRATION1           = 0x1B,
    TDCx_CALIBRATION2           = 0x1C
};

// Commands as in the v2.7 firmware for the MCU interface to the TDC7201
//  (TDC720x is assumed and omitted in the names below; TDC1000 commands are
//   omitted)
enum class TdcUsbCmd : std::uint8_t
{
    LoopPacket = 0x00,
    ReInit = 0x01,
    Start_Continuous_Trigger = 0x04,
    Start_TOF_One_Shot = 0x05,
    Start_TOF_Graph = 0x06,
    End_TOF_Graph = 0x07,
    Stop_Continuous_Trigger = 0x08,
    Firmware_Version_Read = 0x09,
    LED_Toggle = 0x0A,
    MSP430SPI_Config_Read = 0x0B,
    MSP430SPI_Config_Write = 0x0C,
    SPI_Byte_Write = 0x12,
    SPI_Byte_Read = 0x13,
    SPI_Word_Read = 0x14,
    Status_Read = 0x15,
    Status_Write = 0x16,
    Set_ExtOSC_Wakeup_Delay = 0x17,
    Set_Timer_Trigger_Freq = 0x18,
    Set_Xclk_Period = 0x19,
    Read_Xclk_Period = 0x1A,
    Read_Timer_Trigger_Freq = 0x1B,
    Read_ExtOSC_Wakeup_Period = 0x1C,
    MSP430_BSL = 0x1D,
    Reset = 0x1E,
    Set_Double_Resolution = 0x20,
    Clear_Double_Resolution = 0x21,
    SPIAutoIncWrite = 0x22,
    SPIAutoIncRead = 0x23,
    Start_Graph_Delay_Sweep = 0x24,
    Stop_Graph_Delay_Sweep = 0x25,
    Set_Fast_Trig = 0x26,
    Clear_Fast_Trig = 0x27,
    Read_Tdc_Clk_Period = 0x2F,
    Set_Tdc_Clk_Period = 0x30,
    Measure_LT12ns_Enable = 0x31,
    Measure_LT12ns_Disable = 0x32,
    Enable_UART_Stream = 0x33,
    Set_Current_TDC_For_Access = 0x35,
    Read_Current_TDC_For_Access = 0x36,
    Set_TDC_Graph_Select = 0x37,
    Read_TDC_Graph_Select = 0x38,
    Set_SPI_DOUTx = 0x39,
    Read_SPI_DOUTx = 0x3A,
    Set_Device = 0x3B,
    Read_Device = 0x3C
};

struct TdcConfig
{
    // TDCx_CONFIG1 register associated fields:
    TdcCal force_cal = TdcCal::unless_interrupted;
    TdcEdge trig_edge = TdcEdge::rising;
    TdcEdge start_edge = TdcEdge::rising;
    TdcEdge stop_edge = TdcEdge::rising;
    TdcMeasMode meas_mode = TdcMeasMode::mode2;
    // TDCx_CONFIG2 register associated fields:
    int avg_cycles = 1;     // valid = {1, 2, 4, ..., 128}
    int cal2_periods = 10;  // valid = {2, 10, 20, 40}
    int num_stops = 1;      // valid = [1, 5]
    // TDCx_CLOCK_CNTR_STOP_MASK_{L,H} register associated fields:
    int stop_mask_cycles = 0;
    
    bool valid(double ref_clk_hz) const;
};

enum class TdcUnit
{
    tdc1, tdc2
};

// Responses are maximally (32 - 8 + 1) = 25 bytes long, since the response
//  string is 32 bytes fixed and the response begins at byte 8
struct TdcResponse : public std::array<unsigned char, 25>
{
};

class Tdc
{
  public:
    Tdc(const ProgState& prog_state, string_view port_spec, double ref_clk_hz);
    Tdc(const Tdc&) = delete;
    Tdc& operator=(const Tdc&) = delete;

    //picoseconds get_meas();
    picoseconds get_meas_from_stream();

  protected:
    ~Tdc() = default;

  private:
    void setup(const TdcConfig& cfg);
    void setup_unit(TdcUnit unit, const TdcConfig& cfg);
    picoseconds get_avg_meas_mode2();
    picoseconds get_unit_meas_mode2(TdcUnit unit);
    void set_unit(TdcUnit unit);
    //void start_new_meas(TdcUnit unit);
    void start_meas_stream();

    std::uint32_t get_register(TdcRegister);
    void set_register(TdcRegister reg, std::uint32_t val);

    TdcResponse send_command(TdcUsbCmd cmd, std::uint8_t cmd_payload);
    TdcResponse send_command(TdcUsbCmd cmd, std::uint8_t cmd_payload1,
        std::uint8_t cmd_payload2);
    TdcResponse send_command(TdcUsbCmd cmd, ustring_view cmd_payload);

    double ref_clk_hz_;
    optional<TdcUnit> selected_unit_;
    SerialDev serial_dev_;
};

} // end namespace tmon

// Copyright (C) 2021, Lawrence Livermore National Security, LLC.
//  All rights reserved. LLNL-CODE-837067
//
// The Department of Homeland Security sponsored the production of this
//  material under DOE Contract Number DE-AC52-07N427344 for the management
//  and operation of Lawrence Livermore National Laboratory. Contract no.
//  DE-AC52-07NA27344 is between the U.S. Department of Energy (DOE) and
//  Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.
//  See license for disclaimers, notice of U.S. Government Rights and license
//  terms and conditions.

#endif // TDC_H_
