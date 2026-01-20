#include "spmc.h"

// Single-producer/multiple-consumer queue
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

#if defined(UNIT_TEST)
#include <iostream>
#include <random>
#include <boost/thread/scoped_thread.hpp>

int main()
{
    tmon::SpmcQueue<int> q{3};
    for (int i = 1; i <= 10; ++i)
        q.push_back(i);
    assert(q.pop_front(1) == 1);
    assert(q.pop_front(2) == 1);
    assert(q.pop_front(1) == 2);
    assert(q.pop_front(1) == 3);
    assert(q.pop_front(3) == 1);
    assert(q.pop_front(2) == 2);
    assert(q.pop_front(1) == 4);
    assert(q.pop_front(2) == 3);
    assert(q.pop_front(2) == 4);
    assert(q.pop_front(3) == 2);
    assert(q.pop_front(3) == 3);
    assert(q.pop_front(3) == 4);
    for (int i = 5; i <= 10; ++i)
    {
        assert(q.pop_front(1) == i);
        assert(q.pop_front(2) == i);
        assert(q.pop_front(3) == i);
    }
    assert(q.empty(1));
    assert(q.empty(2));
    assert(q.empty(3));
    q.push_back(11);
    assert(!q.empty(1));
    assert(!q.empty(2));
    assert(!q.empty(3));
    assert(q.pop_front(3) == 11);
    assert(q.pop_front(2) == 11);
    assert(q.pop_front(1) == 11);
    assert(q.empty(1));
    assert(q.empty(2));
    assert(q.empty(3));
    std::cerr << "Single-threaded test succeeded." << std::endl;

    // Multi-threaded test:
    constexpr int num_threads{10};
    constexpr int max_count{100};
    tmon::SpmcQueue<int> qmt{num_threads};
    qmt.push_back(0);
    auto thread_fn = [&qmt, ctr = 0](int c) mutable {
        while (ctr != max_count)
        {
            assert(qmt.wait(c, []{ return false; }));
            assert(qmt.pop_front(c) == ctr++);
        }
        std::cerr << "Consumer thread " << c << " done." << std::endl;
        assert(qmt.empty(c));
    };
    std::vector<boost::scoped_thread<>> threads;
    for (int i = 0; i < num_threads; ++i)
        threads.emplace_back(thread_fn, i + 1);
    std::cerr << "Launched " << num_threads << " consumer threads" << std::endl;
    std::mt19937 rng;
    std::uniform_int_distribution<> unif{1, 500};
    for (int i = 1; i < max_count; ++i)
    {
        qmt.push_back(i);
        std::cerr << "Produced value " << i << "/" << max_count << std::endl;
        auto rand_ms = unif(rng);
        boost::this_thread::sleep_for(boost::chrono::milliseconds{rand_ms});
    }
    std::cerr << "Done producing values." << std::endl;
    for (auto& t : threads)
    {
        t.join();
        std::cerr << "Joined consumer thread." << std::endl;
    }
    std::cerr << "Multi-threaded test succeeded." << std::endl;
    std::cerr << "Test succeeded." << std::endl;
    return EXIT_SUCCESS;
}

#endif

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

