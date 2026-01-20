#ifndef PROG_STATE_H_
#define PROG_STATE_H_

#include <exception>
#include <boost/program_options.hpp>

#include "common.h"

// Program State
//  This unit includes common program state (conceptually global across tasks,
//  but not implemented as a global due to the dependency-inversion principle).
//  This state includes the set of configuration parameters parsed from the
//  command line and configuration file, as well as a means of accessing the
//  logging stream.  Since exceptions can be thrown not only from the main
//  thread but from task threads as well, this unit also includes a mechanism to
//  capture and transport exceptions from tasks.  Finally, it maintains the quit
//  flag used to atomically signal all running threads (and interruptible
//  blocking calls) that the application should quit via cooperative shutdown.
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

class ProgState
{
  public:
    using OptMapType = boost::program_options::variables_map;
    using OptValueType = boost::program_options::variable_value;

    ProgState(volatile std::atomic<bool>& quit_flag);
    ~ProgState() = default;
    ProgState(const ProgState&) = delete;
    ProgState& operator=(const ProgState&) = delete;

    LoggerType& get_logger() const
    {
        return logger_;
    }
    
    const OptValueType& get_opt(const std::string& opt_name) const
        /* throws std::out_of_range */
    { 
        return opt_map_.at(opt_name);
    }
    template <typename T>
    const T& get_opt_as(const std::string& opt_name) const
        /* throws std::out_of_range */
    {
        return get_opt_as<T>(opt_name, 0);
    }
    template <typename T>
    const T& get_opt_as(const std::string& opt_name, std::size_t idx) const
        /* throws std::out_of_range */
    {
        const OptValueType& opt_val = get_opt(opt_name);
        if (const T* scalar_val = boost::any_cast<const T>(&opt_val.value());
            scalar_val)
        {
            // Allow a scalar parameter to stand in for a vector parameter
            //  only when the index provided is zero
            if (idx != 0)
                throw std::out_of_range("scalar option but idx nonzero");
            return *scalar_val;
        }
        const auto& param_vec = opt_val.as<std::vector<T>>();
        if (idx >= param_vec.size())
            throw std::out_of_range("Idx out of range for option " + opt_name);
        return param_vec[idx];
    }
    bool has_opt(const std::string& opt_name) const
    { 
        return opt_map_.count(opt_name); 
    }

    void parse_config(int argc, const char* const argv[]);
        /* throws std::runtime_error */

    std::exception_ptr get_exception() const
        { return exception_; }
    std::string get_exception_task() const
        { return exception_task_; }
    void set_exception(std::exception_ptr&& e) const
        { exception_ = std::move(e); }
    void set_exception_task(std::string task_name) const
        { exception_task_ = std::move(task_name); }
    bool should_quit() const
        { return quit_flag_ || exception_; }

  private:
    OptMapType opt_map_;
    mutable LoggerType logger_;
    mutable std::exception_ptr exception_;
    mutable std::string exception_task_;
    volatile std::atomic<bool>& quit_flag_;
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

#endif // PROG_STATE_H_
