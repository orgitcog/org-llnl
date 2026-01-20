#include <algorithm>
#include "clock.h"
#include "task.h"
#include "task_container.h"

// Task Container
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

TaskContainer::TaskContainer(string_view name, const ProgState& prog_state) 
    : TaskBase{name, prog_state}, tasks_{}
{
}

TaskContainer::~TaskContainer()
{
    TM_LOG(debug) << "Destroying task container " << get_name();
    if (!is_stopped())
        stop();
    // Ensure that tasks are destroyed in reverse order of insertion
    //  (without the clear here, they'd be destroyed in undefined order)
    if (!tasks_.empty())
        clear();
}

bool TaskContainer::add_task(std::unique_ptr<TaskBase> task)
{
    assert(task && "Null ptr passed to add_task");
    auto name_match_fn = [&task](TaskCont::const_reference t2) { 
        assert(t2);
        return task->get_name() == t2->get_name(); };
    if (std::any_of(tasks_.cbegin(), tasks_.cend(), name_match_fn))
    {
        TM_LOG(error) << "Task '" << task->get_name() << "' not added; name "
            "not unique in list";
        return false; // Task name must be unique
    }
    TM_LOG(debug) << "Task '" << task->get_name() << "' added to container";
    tasks_.push_back(std::move(task));
    return true;
}

void TaskContainer::start_iface_hook()
{
    assert(is_stopped());
    TM_LOG(debug) << "Starting all tasks for container " << get_name();
    set_stop_flag(false);
    start_hook();
    for (auto& t : tasks_)
    {
        assert(t);
        TM_LOG(debug) << "Container " << get_name() << " starting task "
            << t->get_name();
        t->start();
    }
    TM_LOG(debug) << "Container " << get_name() << " started all tasks";
}

void TaskContainer::start_hook()
{
}

void TaskContainer::stop_iface_hook()
{
    assert(!is_stopped());
    TM_LOG(debug) << "Stopping all tasks for container " << get_name();
    set_stop_flag(true);
    stop_hook();
    for (auto& t : tasks_)
    {
        assert(t);
        if (t->is_stopped())
        {
            TM_LOG(debug) << "Container " << get_name() << " skipping task "
                << t->get_name() << " stop; already stopped";
        }
        else
        {
            TM_LOG(debug) << "Container " << get_name() << " stopping task "
                << t->get_name();
            t->stop();
        }
    }
    TM_LOG(debug) << "Container " << get_name() << " stopped all tasks";
}

void TaskContainer::stop_hook()
{
}

void TaskContainer::clear()
{
    TM_LOG(debug) << "Destroying all tasks for container " << get_name();
    for (auto i = tasks_.rbegin(); i != tasks_.rend(); ++i)
    {
        assert(*i);
        TM_LOG(debug) << "Container " << get_name() << " destroying task "
            << (*i)->get_name();
        i->reset();
    }
    tasks_.clear();
    TM_LOG(debug) << "Container " << get_name() << " destroyed all tasks";
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

