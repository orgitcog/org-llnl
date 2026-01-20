#ifndef TASK_H_
#define TASK_H_

#include <boost/thread/scoped_thread.hpp>
#include "common.h"
#include "prog_state.h"

// Task
//  This unit provides a base interface for tasks that can be started and
//  stopped, with associated information such as names and run states.  A
//  concrete version that also has a managed thread is here implemented.
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

class TaskInterface
{
  public:
    virtual ~TaskInterface() = default;

    void start();
    void stop();

  protected:
    virtual void start_iface_hook() = 0;
    virtual void stop_iface_hook() = 0;
};

class TaskBaseImpl
{
  public:
    TaskBaseImpl(string_view name, const ProgState& prog_state);
    virtual ~TaskBaseImpl() = default;
    TaskBaseImpl(const TaskBaseImpl&) = delete;
    TaskBaseImpl& operator=(const TaskBaseImpl&) = delete;
    TaskBaseImpl(TaskBaseImpl&&) = default;

    LoggerType& get_logger() const { return prog_state_.get_logger(); }
    std::string get_name() const { return name_; }
    bool should_quit() const { return prog_state_.should_quit() || stopped_; }
    bool is_stopped() const { return stopped_; }

  protected:
    const ProgState::OptValueType& get_opt(const std::string& opt_name) const
        /* throws std::out_of_range */
    {
        return prog_state_.get_opt(opt_name);
    }
    template <typename T>
    const T& get_opt_as(const std::string& opt_name) const
        /* throws std::out_of_range */
    {
        return prog_state_.get_opt_as<T>(opt_name);
    }
    template <typename T>
    const T& get_opt_as(const std::string& opt_name, std::size_t idx) const
        /* throws std::out_of_range */
    {
        return prog_state_.get_opt_as<T>(opt_name, idx);
    }
    bool has_opt(const std::string& opt_name) const
    {
        return prog_state_.has_opt(opt_name);
    }
    void set_exception(std::exception_ptr&& e, std::string task_name) const;
    void set_stop_flag(bool stopped)
    {
        stopped_ = stopped;
    }

  private:
    std::string name_;
    const ProgState& prog_state_;
    std::atomic<bool> stopped_;
};

class TaskBase : public TaskInterface, public TaskBaseImpl
{
    using TaskBaseImpl::TaskBaseImpl;
};

class Task : public TaskBase
{
  public:
    Task(string_view name, const ProgState& prog_state);
    ~Task() override;
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    Task(Task&&) = default;

  protected:
    void start_iface_hook() final;
    void stop_iface_hook() final;

    virtual void start_hook();
    virtual void run() = 0;
    virtual void stop_hook() = 0;

  private:
    boost::scoped_thread<> thread_;
};

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

#endif // TASK_H_
