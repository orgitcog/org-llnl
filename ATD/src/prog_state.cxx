#include "prog_state.h"
#include <fstream>
#include <iostream>
#include "alert_msg.h"
#include "param.h"

// prog_state - Common program state (configuration options, etc.)
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
static constexpr char tm_prog_name[] = "Assured Timing Detector";
static constexpr int tm_major_ver = 1;
static constexpr int tm_minor_ver = 0;
static constexpr int tm_patch_ver = 0;

ProgState::ProgState(volatile std::atomic<bool>& quit_flag)
    : opt_map_{}, logger_{}, exception_{}, exception_task_{},
        quit_flag_{quit_flag}
{
}

// Parses configuration parameters passed through the specified command-line
//  arguments (as well as any referenced configuration file) and stores the
//  options map within the program state
//
// (Note: In keeping with convention, this function will exit with success after
//  displaying parameter options if the "help" option is passed)
void ProgState::parse_config(int argc, const char* const argv[])
    /* throws std::runtime_error */
{
    namespace prog_opt = boost::program_options;

    std::string config_file;
    prog_opt::options_description opt_desc("Supported options");
    opt_desc.add_options()
        ("help,h", "show help message")
        ("config-file,c", prog_opt::value<std::string>(
            &config_file)->default_value("tmonitor.cfg"),
            "path to configuration file");

    prog_opt::options_description opt_desc_log{"Logging options"};
    opt_desc_log.add_options()
        ("log.verbosity,v", prog_opt::value<int>()->default_value(4),
            "relative verbosity of log output (max=4)")
        ("log.console_verbosity", prog_opt::value<int>()->default_value(4),
            "relative verbosity of console log output (max=4)")
        ("log.file_verbosity", prog_opt::value<int>()->default_value(4),
            "relative verbosity of file log output (max=4)");
    opt_desc.add(opt_desc_log);

    prog_opt::options_description opt_desc_alert{"Alerting options"};
    opt_desc_alert.add_options()
        ("alert.level.external_spoof_det",
            prog_opt::value<Alert::Level>()->default_value(Alert::Level::red),
            "alert level for external spoofing detection (e.g., by receiver)")
        ("alert.level.interference",
            prog_opt::value<Alert::Level>()->default_value(
                Alert::Level::yellow),
            "alert level for RF interference (low carrier/noise)")
        ("alert.level.invalid_tow",
            prog_opt::value<Alert::Level>()->default_value(
                Alert::Level::yellow),
            "alert level for invalid GNSS time-of-week")
        ("alert.level.pos_error",
            prog_opt::value<Alert::Level>()->default_value(
                Alert::Level::yellow),
            "alert level for position error")
        ("alert.level.lost_clock",
            prog_opt::value<Alert::Level>()->default_value(
                Alert::Level::yellow),
            "alert level for early loss of clock input")
        ("alert.level.lost_fix",
            prog_opt::value<Alert::Level>()->default_value(
                Alert::Level::yellow),
            "alert level for lost GNSS fix")
        ("alert.level.phase_anomaly",
            prog_opt::value<Alert::Level>()->default_value(
                Alert::Level::red),
            "alert level for phase anomaly")
        ("alert.level.lower_phase_anomaly",
            prog_opt::value<Alert::Level>()->default_value(
                Alert::Level::yellow),
            "alert level for lower-bound phase anomaly")
        ("alert.expiry.yellow", prog_opt::value<int>()->default_value(60000),
            "expiry time (in ms) of yellow alerts")
        ("alert.expiry.red", prog_opt::value<int>()->default_value(60000),
            "expiry time (in ms) of red alerts");
    opt_desc.add(opt_desc_alert);

    prog_opt::options_description opt_desc_cal{"Calibration options"};
    opt_desc_cal.add_options()
        ("cal.neg_constrain_thresh_pct",
            prog_opt::value<double>()->default_value(-99.9),
            "negative threshold to constrain calibration (in % vs. original)")
        ("cal.pos_constrain_thresh_pct",
            prog_opt::value<double>()->default_value(1e5),
            "positive threshold to constrain calibration (in % vs. original)")
        ("cal.duration",
            prog_opt::value<int>()->default_value(3600),
            "duration of the calibration period (in s)")
        ("cal.enabled",
            prog_opt::value<bool>()->default_value(true),
            "true to enable calibration, false to disable")
        ("cal.reference_quieter_mults",
            prog_opt::value<VectorParam<double>>()->default_value({1e-4, 1e5}),
            "list of multipliers used to assess whether a potential reference "
                "clock is sufficiently quieter")
        ("cal.reference_tau_ms",
            prog_opt::value<VectorParam<double>>()->default_value(
                {1e3, 3600e3}),
            "list of integration periods (in ms) for establishing a reference "
                "pair")
        ("cal.neg_reject_thresh_pct",
            prog_opt::value<double>()->default_value(-99.99),
            "negative threshold to reject calibration (in % vs. original)")
        ("cal.pos_reject_thresh_pct",
            prog_opt::value<double>()->default_value(1e6),
            "positive threshold to reject calibration (in % vs. original)")
        ("cal.min_tau_wfm",
            prog_opt::value<double>()->default_value(0.5),
            "minimum tau_0 for which to attempt WFM calibration")
        ("cal.tau_mults", prog_opt::value<VectorParam<int>>()->default_value(
                {1, 3, 5, 7, 10}),
            "list of integration periods (multiples of tau_0) for calibration");
    opt_desc.add(opt_desc_cal);

    prog_opt::options_description opt_desc_det{"Detection options"};
    opt_desc_det.add_options()
        ("det.thresh_sigma,k", prog_opt::value<double>()->default_value(1.96),
            "detection threshold sigma")
        ("det.lower_thresh_sigma", prog_opt::value<double>()->default_value(0),
            "lower detection threshold sigma")
        ("det.alert_ignore_duration",
            prog_opt::value<int>()->default_value(60),
            "duration (in s) to ignore alerts after startup")
        ("det.alert_ignore_duration_postcal",
            prog_opt::value<int>()->default_value(30),
            "duration (in s) to ignore alerts after apply calibration")
        ("det.meas_noise", prog_opt::value<double>()->default_value(3e-17),
            "detection meas. noise variance assumed")
        ("det.initial_cov", prog_opt::value<double>()->default_value(1e-40),
            "initial state covariance")
        ("det.stat_logfile", prog_opt::value<std::string>()->default_value(
            "log/wssr.log"),
            "path to logfile for detection statistics ('disable' for none)")
        ("det.window_len", prog_opt::value<int>()->default_value(100),
            "detection window length");
    opt_desc.add(opt_desc_det);

    prog_opt::options_description opt_desc_gnss{"GNSS options"};
    opt_desc_gnss.add_options()
        ("gnss.ecef_x", prog_opt::value<std::vector<long>>(),
            "Fixed ECEF x-coordinate (cm)")
        ("gnss.ecef_y", prog_opt::value<std::vector<long>>(),
            "Fixed ECEF y-coordinate (cm)")
        ("gnss.ecef_z", prog_opt::value<std::vector<long>>(),
            "Fixed ECEF z-coordinate (cm)")
        ("gnss.ecef_x_tol", prog_opt::value<long>()->default_value(10000),
            "ECEF x-coordinate deviation tolerance (cm)")
        ("gnss.ecef_y_tol", prog_opt::value<long>()->default_value(10000),
            "ECEF y-coordinate deviation tolerance (cm)")
        ("gnss.ecef_z_tol", prog_opt::value<long>()->default_value(10000),
            "ECEF z-coordinate deviation tolerance (cm)")
        ("gnss.ecef_log",
            prog_opt::value<bool>()->default_value(false),
            "True to log ECEF coordinates; false otherwise")
        ("gnss.serial_port",
            prog_opt::value<std::vector<std::string>>()->default_value(
                {"COM1|9600,8,N,1,none"}, "COM1|9600,8,N,1,none"),
            "GNSS receiver serial port (and optional spec)")
        ("gnss.overall_class",
            prog_opt::value<std::vector<std::string>>()->default_value(
                {"gpsdo_indoor"}, ""),
            "Clock class for the overall time source")
        ("gnss.internal_class",
            prog_opt::value<std::vector<std::string>>()->default_value(
                {"vctcxo"}, ""),
            "Clock class for the internal time source")
        ("gnss.gnss_class",
            prog_opt::value<std::vector<std::string>>()->default_value(
                {"gpsdo_best"}, ""),
            "Clock class for the GNSS time source")
        ("gnss.extint0_class",
            prog_opt::value<std::vector<std::string>>()->default_value(
                {"ocxo_good"}, ""),
            "Clock class for the external time source 0")
        ("gnss.extint1_class",
            prog_opt::value<std::vector<std::string>>()->default_value(
                {"ocxo_good"}, ""),
            "Clock class for the external time source 1");
    opt_desc.add(opt_desc_gnss);

    prog_opt::options_description opt_desc_tdc{"TDC options"};
    opt_desc_tdc.add_options()
        ("tdc.serial_port", prog_opt::value<std::string>()->default_value(
            "COM1|38400,8,N,1,none"),
            "TDC serial port and specification");
    opt_desc.add(opt_desc_tdc);

    prog_opt::options_description opt_desc_tcp_alert{"TCP alerter options"};
    opt_desc_tcp_alert.add_options()
        ("tcpalert.port", prog_opt::value<unsigned short>()->default_value(
            8123),
            "TCP alerter port number");
    opt_desc.add(opt_desc_tcp_alert);

    prog_opt::options_description opt_desc_ws{"WebSocket server options"};
    opt_desc_ws.add_options()
        ("wsserver.port", prog_opt::value<unsigned short>()->default_value(
            8118),
            "WebSocket server port number");
    opt_desc.add(opt_desc_ws);

    prog_opt::options_description opt_desc_test{"Testing options (replay)"};
    opt_desc_test.add_options()
        ("test.replay.fixed_delay", prog_opt::value<int>()->default_value(1000),
            "Fixed delay (in us) between replay steps")
        ("test.replay.use_fixed_delay",
            prog_opt::value<bool>()->default_value(false),
            "True to use fixed_delay between msgs; false to use timestamps")
        ("test.replay.min_delay", prog_opt::value<int>()->default_value(1000),
            "Minimum delay (in us) between replay steps")
        ("test.replay.full_buffer_delay",
            prog_opt::value<int>()->default_value(5000),
            "Delay (in ms) after replay buffer is full");
    opt_desc.add(opt_desc_test);

    prog_opt::options_description opt_desc_test_tmp{
        "Testing options"};
    opt_desc_test_tmp.add_options()
        ("tmp.canary_file", prog_opt::value<std::string>()->default_value(
            "tmonitor.quit"),
            "Canary filename (quits when file exists)")
        ("tmp.max_iter", prog_opt::value<int>()->default_value(-1),
            "Maximum number of 1s iter of main loop")
      #if defined(ENABLE_TEST_INJECT)
        ("tmp.inject_file", prog_opt::value<std::string>()->default_value(
            "tmonitor.inject"),
            "Inject file path for testing")
        ("tmp.inject_gnss_idx", prog_opt::value<int>()->default_value(0),
            "Inject GNSS task index for testing")
      #endif // defined(ENABLE_TEST_INJECT)
        ("tmp.replay_file", prog_opt::value<std::string>()->default_value(
            "test_timemsgs2.out"),
            "Test file for ClockReplay")
        ("tmp.timemsg_file", prog_opt::value<std::string>()->default_value(
            "test_timemsgs.out"),
            "Test file for FileWriterPseudoAlg")
        ("tmp.ubx_file", prog_opt::value<std::string>()->default_value(
            "../include/thirdparty/gpsd/ublox-8-time.log"),
            "Test file for UbxFile");
    opt_desc.add(opt_desc_test_tmp);

    prog_opt::variables_map opt_map;
    prog_opt::store(prog_opt::parse_command_line(argc, argv, opt_desc), 
        opt_map);
    prog_opt::notify(opt_map);

    if (opt_map.count("help"))
    {
        // Show help message for command-line options
        std::cout << tm_prog_name << " - version " << tm_major_ver << "."
            << tm_minor_ver << "." << tm_patch_ver << "\n";
        std::cout << opt_desc << "\n";
        std::exit(EXIT_SUCCESS);
    }

    if (!config_file.empty())
    {
        // Load program options from a specified configuration file
        std::ifstream ifs(config_file);
        if (!ifs)
        {
            throw std::runtime_error("Unable to open config file '" +
                config_file + "'");
        }
        auto parsed_opts = prog_opt::parse_config_file(ifs, opt_desc, false);
        prog_opt::store(parsed_opts, opt_map);
        prog_opt::notify(opt_map);
    }

    opt_map_ = std::move(opt_map);
}

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

