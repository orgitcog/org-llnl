#ifndef SPMC_H_
#define SPMC_H_

#include "common.h"
#include <deque>
#include <vector>

// Single-producer/multiple-consumer queue
//  Concurrent storage queue for multiple consumers.  Pushing an element makes
//    it visible to all consumers; elements are popped after all
//    consumers have accessed them
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

template <typename T>
class SpmcQueue
{
  public:
    SpmcQueue();
    SpmcQueue(int num_consumers);
    ~SpmcQueue();

    int add_consumer();

    // Cancel any blocking waits on this queue (by signaling conditional)
    void cancel();

    // Listener is a user-friendly means of adding a new consumer and bundling
    //  the consumer index with a reference to the queue, alongside an adaptor
    //  of the queue interface exposing only the consumer member functions
    class Listener;
    Listener get_new_listener();

    // Pushes a copy of an element to the queue, with a tag indicating that
    //  none of the consumers have yet popped it
    void push_back(T x);

    // Returns a copy of the front element in the queue if there may be other
    //  consumers yet to read it; otherwise, pops it and returns by move
    optional<T> pop_front(int consumer);

    // Whether or not the queue is empty for the specified consumer (which may,
    //  but does not necessarily, imply that the queue is empty altogether)
    [[nodiscard]] bool empty(int consumer) const;

    // Reserve capacity in the queue for the specified number of entries
    void reserve(std::size_t sz);

    // Waits until data is available for the specified consumer (i.e., empty
    //  would return false) or the provided predicate is satisfied; returns true
    //  if data is available
    template <typename Predicate>
    bool wait(int consumer, Predicate fn) const;

  private:
    struct CountedDatum
    {
        T data;
        int potential_consumers;
    };
    int num_consumers_;
    mutable mutex mutex_;
    mutable condition_variable cond_;
    std::deque<CountedDatum> cont_;
    std::vector<int> head_for_consumer_;

    [[nodiscard]] bool unsynchronized_empty(int consumer) const;
};

} // end namespace tmon

#include "../src/spmc.txx"

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

#endif // SPMC_H_

