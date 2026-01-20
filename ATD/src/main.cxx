#include <atomic>
#include <cassert>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <boost/filesystem.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/expressions/formatters/date_time.hpp>
#include <boost/log/expressions/formatters/named_scope.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/file.hpp>

#if (__has_include("wincon.h"))
#include <windef.h>
#include <winbase.h>
#include <wincon.h>
#endif

#include "alerter.h"
#include "clock_replay.h"
#include "common.h"
#include "detector.h"
#include "prog_state.h"
#include "stream_clock.h"
#include "task_container.h"
#include "tcp_alert_handler.h"
#include "ubx_file_reader.h"
#include "ubx_serial.h"
#include "ws_server.h"

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

namespace
{
    volatile std::atomic<bool> quit_flag{false};
    ProgState prog_state{quit_flag};
    struct Loggers
    {
        boost::shared_ptr<boost::log::sinks::synchronous_sink<
            boost::log::sinks::basic_text_ostream_backend<char>>> console_log;
        boost::shared_ptr<boost::log::sinks::synchronous_sink<
            boost::log::sinks::text_file_backend>> file_log;
    };

    LoggerType& get_logger()
    {
        return prog_state.get_logger();
    }
}

void display_notice()
{
    auto notice_text = R"(
    Copyright (C) 2021, Lawrence Livermore National Security, LLC.
    All rights reserved. LLNL-CODE-837067

    The Department of Homeland Security sponsored the production of this
    material under DOE Contract Number DE-AC52-07N427344 for the management
    and operation of Lawrence Livermore National Laboratory. Contract no.
    DE-AC52-07NA27344 is between the U.S. Department of Energy (DOE) and
    Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.
    See license for disclaimers, notice of U.S. Government Rights and license
    terms and conditions.
    )";
    std::cerr << notice_text << std::endl;
}

BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", SeverityType)

Loggers setup_loggers()
{
    using namespace boost;
    Loggers ret;

    // Add a console log (to std::clog)
    ret.console_log = log::add_console_log(std::clog, log::keywords::format =
        (log::expressions::stream << 
            log::expressions::attr<unsigned int>("LineID") << ": [" 
            << log::expressions::format_date_time<boost::posix_time::ptime>(                "TimeStamp", "%Y-%m-%d %H:%M:%S")
            << "] <" << log::trivial::severity
            << "> " << log::expressions::smessage));

    // Add a file log that rotates each day at midnight
    ret.file_log = log::add_file_log(
        log::keywords::file_name = "log/tmonitor_%Y-%m-%d_%H_%M_%S.log",
        log::keywords::time_based_rotation =
            log::sinks::file::rotation_at_time_point(0, 0, 0),
        log::keywords::format = (log::expressions::stream
            << log::expressions::attr<unsigned int>("LineID") << ": [" 
            << log::expressions::attr<boost::posix_time::ptime>("TimeStamp")
            << "] <" << log::trivial::severity << "> " 
            << log::expressions::smessage
            << log::expressions::format_named_scope("Scope",
                log::keywords::format = " [in %n]")));

    log::add_common_attributes();
    log::core::get()->add_global_attribute("Scope",
        log::attributes::named_scope{});
    return ret;
}

void handle_signal(int sig) noexcept
{
    switch (sig)
    {
        case SIGINT:
        case SIGTERM:
            quit_flag = true;
        default:
            break;
    }
}

//BOOL WINAPI win_handle_console_signal(DWORD sig)
int __stdcall win_handle_console_signal(unsigned long sig)
{
    //TM_LOG(warning) << "Got Windows console signal " << sig;
    std::cerr << "Got Windows console signal " << sig << std::endl;
    switch (sig)
    {
        case 0: // CTRL_C_EVENT
        case 1: // CTRL_BREAK_EVENT
        case 2: // CTRL_CLOSE_EVENT
            quit_flag = true;
            return 1; // TRUE = handled signal; don't call other handlers
        default:
            return 0; // FALSE = not handled; call other handlers
    }
}

void setup_signal_handlers()
{
    assert(quit_flag.is_lock_free() && "Need lock-free atomic for quit flag");
    std::signal(SIGINT, &handle_signal);
    
#if (__has_include("wincon.h"))
    // Since on Windows, the signal handler installed above may not suffice to
    //  catch Ctrl-C on console, install another handler
    assert(SetConsoleCtrlHandler(&win_handle_console_signal, TRUE));
    TM_LOG(debug) << "Installed Windows signal handler";

    // And on MSYS2, even that doesn't seem to work; Ctrl-C will abruptly
    //  terminate the process there
#endif
}

int main(int argc, char* argv[])
{
try
{
    display_notice();

    Loggers loggers = setup_loggers();
    setup_signal_handlers();
    // Disable sync with stdio (printf-family functions not used)
    std::ios_base::sync_with_stdio(false);

    // Log invocation arguments
    std::ostringstream args_oss;
    for (int i = 0; i < argc; ++i)
        args_oss << argv[i] << " ";
    TM_LOG(info) << "Invoked as " << args_oss.str();

    // Handle program options (parameters + configuration file)
    prog_state.parse_config(argc, argv);
    TM_LOG(info) << "Configuration processed";

    auto apply_verbosity = [](auto& logger, auto sev_thresh){
        auto set_min_severity = [](auto& logger, auto sev){
            logger.set_filter(boost::log::trivial::severity >= sev);};
        switch (sev_thresh)
        {
            case 4: // maximum verbosity (process all log severity levels)
                break; // no filter in this case
            case 3: // medium verbosity (log severity >= debug)
                set_min_severity(logger, boost::log::trivial::debug);
                break;
            case 2: // low verbosity (log severity >= info)
                set_min_severity(logger, boost::log::trivial::info);
                break;
            case 1: // very low verbosity (log severity >= warning)
                set_min_severity(logger, boost::log::trivial::warning);
                break;
            case 0: // minimum verbosity (log severity >= error)
                set_min_severity(logger, boost::log::trivial::error);
                break;
            default:
                TM_LOG(warning) << "Unknown log verbosity ignored";
                break;
        }
    };
    apply_verbosity(*boost::log::core::get(),
        prog_state.get_opt_as<int>("log.verbosity"));
    apply_verbosity(*loggers.console_log,
        prog_state.get_opt_as<int>("log.console_verbosity"));
    apply_verbosity(*loggers.file_log,
        prog_state.get_opt_as<int>("log.file_verbosity"));

    TaskContainer task_cont{"Main", prog_state};

    auto alerter_task = std::make_unique<Alerter>("Alerter", prog_state);

    const unsigned short tcp_port = prog_state.get_opt_as<unsigned short>(
        "tcpalert.port");
    Alerter::Listener tcp_alert_listener = alerter_task->register_listener();
    auto tcp_alert_handler_task = std::make_unique<TCPAlertHandler>(
        "TCPAlertHandler", prog_state, std::move(tcp_alert_listener), tcp_port);

    auto det_task = std::make_unique<Detector>("Detector", prog_state,
        *alerter_task);
    const Detector& det = *det_task;

    std::string test_file = prog_state.get_opt_as<std::string>("tmp.ubx_file");
    std::unique_ptr<UbxFileReader> test_file_reader{};
    if (!test_file.empty() && test_file != "disable")
    {
        const std::size_t gnss_idx{0};
        test_file_reader = std::make_unique<UbxFileReader>("UbxFile",
            prog_state, *det_task, test_file, gnss_idx);
    }

    auto replay_file = prog_state.get_opt_as<std::string>("tmp.replay_file");
    std::unique_ptr<ClockReplay> clock_replay_task{};
    if (!replay_file.empty() && replay_file != "disable")
    {
        auto test_file_clock = std::make_unique<FileClock>(*det_task,
            ClockDesc{"Replay", "FileTest",
                GaussMarkovModel::ClockClass::pseudo}, replay_file);
        clock_replay_task = std::make_unique<ClockReplay>("ClockReplay",
            prog_state, *det_task, std::move(test_file_clock));
    }

    const auto& port_specs_param = prog_state.get_opt("gnss.serial_port");
    const auto& port_specs = port_specs_param.as<std::vector<std::string>>();
    std::vector<std::unique_ptr<UbxSerial>> serial_tasks{};
    TM_LOG(info) << "Ready to process " << port_specs.size()
        << " serial port specs";
    for (std::size_t gnss_idx = 0; gnss_idx < port_specs.size(); ++gnss_idx)
    {
        auto& port_spec = port_specs[gnss_idx];
        TM_LOG(info) << "Processing serial port spec #" << gnss_idx << ": "
            << port_spec;
        if (!port_spec.empty() && (port_spec != "disable"))
        {
            serial_tasks.push_back(std::make_unique<UbxSerial>("UbxSerial[" +
                std::to_string(gnss_idx) + "]",
                prog_state, *det_task, port_spec, gnss_idx));
        }
    }

    const unsigned short ws_port_num = prog_state.get_opt_as<unsigned short>(
        "wsserver.port");
    auto ws_server_task = std::make_unique<WsServer>("WsServer", prog_state,
        *det_task, ws_port_num);

    task_cont.add_task(std::move(det_task));
    task_cont.add_task(std::move(tcp_alert_handler_task));
    task_cont.add_task(std::move(alerter_task));
    if (test_file_reader)
        task_cont.add_task(std::move(test_file_reader));
    if (clock_replay_task)
        task_cont.add_task(std::move(clock_replay_task));
    for (auto&& serial_task : serial_tasks)
        task_cont.add_task(std::move(serial_task));
    task_cont.add_task(std::move(ws_server_task));

    task_cont.start();

    std::cerr << "Waiting..." << std::endl;
    int i = 0;
    boost::filesystem::path canary_path{prog_state.get_opt_as<std::string>(
        "tmp.canary_file")};
    int max_iter = prog_state.get_opt_as<int>("tmp.max_iter");
    while(!prog_state.should_quit() && ((max_iter < 0) || (i++ < max_iter)))
    {
        boost::this_thread::sleep_for(boost::chrono::seconds(1));
        std::cerr << "Main loop iteration #" << i << std::endl;
        if (boost::filesystem::exists(canary_path))
        {
            TM_LOG(info) << "Processing quit request via canary file";
            quit_flag = true;
        }
        if (det.all_clocks_done() && !det.is_processing())
        {
            TM_LOG(info) << "All clocks + detector done; shutting down";
            quit_flag = true;
        }
    }
    std::cerr << "Quit flag is " << quit_flag << std::endl;
    quit_flag = true;
    task_cont.stop();

    // Run all task destructors before destroying logger
    task_cont.clear();

    // Check for stored exception from any launched task
    std::exception_ptr eptr = prog_state.get_exception();
    if (eptr)
    {
        TM_LOG(error) << "Exception thrown by launched task "
            << prog_state.get_exception_task();
        std::rethrow_exception(eptr);
    }

    TM_LOG(info) << "Program exiting with success";
    return EXIT_SUCCESS;
}
catch(const std::exception& e)
{
    std::cerr << "FATAL ERROR: " << e.what() << std::endl;
    TM_LOG(error) << "Program exiting with FATAL ERROR: " << e.what();
}
catch(...)
{
    std::cerr << "UNSPECIFIED FATAL ERROR" << std::endl;
    TM_LOG(error) << "Program exiting with UNSPECIFIED FATAL ERROR";
}
}

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

