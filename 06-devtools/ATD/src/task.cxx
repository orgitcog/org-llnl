#include "task.h"

// task
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

Task::Task(string_view name, const ProgState& prog_state)
    : TaskBase{name, prog_state}, thread_{}
{
    TM_LOG(debug) << "Created task " << name;
}

Task::~Task()
{
    TM_LOG(debug) << "Destroying task " << get_name();
    if (thread_.joinable())
    {
        TM_LOG(warning) << "Task " << get_name() << 
            " will be automatically stopped; stop was not called";
    }
}

TaskBaseImpl::TaskBaseImpl(string_view name, const ProgState& prog_state)
    : name_{name}, prog_state_{prog_state}, stopped_{true}
{
}

void TaskBaseImpl::set_exception(std::exception_ptr&& e,
    std::string task_name) const
{
    if (prog_state_.get_exception())
    {
        TM_LOG(error) << "FATAL ERROR: Multiple exceptions in flight";
        std::unexpected();
    }
    prog_state_.set_exception(std::move(e));
    prog_state_.set_exception_task(std::move(task_name));
}

void TaskInterface::start()
{
    start_iface_hook();
}

void Task::start_iface_hook()
{
    assert(is_stopped());
    set_stop_flag(false);
    TM_LOG(debug) << "Starting task " << get_name();
    start_hook();
    TM_LOG(debug) << "Starting thread for task " << get_name();
    auto thread_main = [](Task* t){
        try
        {
            t->run(); 
            auto get_logger = [t]() -> LoggerType& {
                return const_cast<const Task*>(t)->get_logger(); };
            TM_LOG(debug) << "Task " << t->get_name() << " has finished";
        }
        catch(...)
        {
            t->set_exception(std::current_exception(), t->get_name());
        }};
    thread_ = std::move(boost::scoped_thread<>{thread_main, this});
    TM_LOG(debug) << "Started task " << get_name();
}

void Task::start_hook()
{
}

void TaskInterface::stop()
{
    stop_iface_hook();
}

void Task::stop_iface_hook()
{
    assert(!is_stopped());
    set_stop_flag(true);
    TM_LOG(debug) << "Stopping task " << get_name();
    // Notify derived task of request to stop so it can interrupt any
    //  blocking operations
    stop_hook();
    TM_LOG(debug) << "Sent stop request to task " << get_name();
    if (!thread_.joinable())
    {
        TM_LOG(warning) << "Skipping join for stop request for task "
            << get_name() << "; already stopped";
    }
    else
    {
        TM_LOG(debug) << "Joining thread to stop task " << get_name();
        thread_.join();
    }
    TM_LOG(debug) << "Stopped task " << get_name();
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

