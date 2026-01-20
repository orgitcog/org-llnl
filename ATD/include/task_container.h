#ifndef TASK_CONTAINER_H_
#define TASK_CONTAINER_H_

#include <vector>
#include "common.h"
#include "prog_state.h"
#include "task.h"

// Task Container
//  This unit implements a container for subtasks that can communicate requests
//  to start or stop operation, and ensures destruction in the reverse order of
//  insertion.  Subtasks must have a unique name within this container.
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

class TaskContainer : public TaskBase
{
  public:
    TaskContainer(string_view name, const ProgState& prog_state);
    ~TaskContainer() override;
    TaskContainer(const TaskContainer&) = delete;

    bool add_task(std::unique_ptr<TaskBase> task);
    void clear();

  protected:
    void start_iface_hook() final;
    void stop_iface_hook() final;

    virtual void start_hook();
    virtual void stop_hook();

  private:
    using TaskCont = std::vector<std::unique_ptr<TaskBase>>;
    TaskCont tasks_;
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

#endif // TASK_CONTAINER_H_
