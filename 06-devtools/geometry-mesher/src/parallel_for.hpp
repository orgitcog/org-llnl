#pragma once

#include <thread>

#include "timer.hpp"

struct threadpool {

  std::vector< uint64_t > partition(uint64_t n) {
    uint64_t quotient = n / num_threads;
    uint64_t remainder = n % num_threads;

    std::vector< uint64_t > blocks(num_threads + 1);
    blocks[0] = 0;
    for (int i = 1; i < num_threads + 1; i++) {
      if (remainder > 0) {
        blocks[i] = blocks[i-1] + quotient + 1;
        remainder--;
      } else {
        blocks[i] = blocks[i-1] + quotient;
      }
    }
    return blocks;
  }

  threadpool(int n) : print_timings(false), num_threads(n), stopwatch(n) {}

  template < typename lambda >
  void parallel_for(uint64_t n, const lambda & f) {

    std::vector< uint64_t > blocks = partition(n);

    for (int tid = 0; tid < num_threads; tid++) {
      threads.push_back(
        std::thread([&, tid]() {
          stopwatch[tid].start();
          for (uint64_t i = blocks[tid]; i < blocks[tid+1]; i++) {
            f(i, tid);
          }
          stopwatch[tid].stop();
        })
      );
    }

    for (int i = 0; i < num_threads; i++) {
      threads[i].join();
      if (print_timings) {
        std::cout << "thread " << i << ": " << stopwatch[i].elapsed() << std::endl;
      }
    }

    threads.clear();
  }

  template < typename lambda >
  void parallel_for(uint64_t ni, uint64_t nj, const lambda & f) {
    for (int tid = 0; tid < num_threads; tid++) {
      threads.push_back(
        std::thread([&](uint64_t i0) {
          for (uint64_t i = i0; i < ni; i += num_threads) {
            for (uint64_t j = 0; j < nj; j++) {
              f(i, j);
            }
          }
        }, tid)
      );
    }

    for (int i = 0; i < num_threads; i++) {
      threads[i].join();
      if (print_timings) {
        std::cout << "thread " << i << ": " << stopwatch[i].elapsed() << std::endl;
      }
    }

    threads.clear();
  }

  template < typename lambda >
  void parallel_for(uint64_t ni, uint64_t nj, uint64_t nk, const lambda & f) {
    // handling the single-threaded case this way avoids using
    // std::thread entirely, which lets parallel_for invoke 
    // callables from python (through nanobind) without GIL deadlock.
    if (num_threads == 1) {
      stopwatch[0].start();
      for (uint64_t i = 0; i < ni; i++) {
        for (uint64_t j = 0; j < nj; j++) {
          for (uint64_t k = 0; k < nk; k++) {
            f(i, j, k);
          }
        }
      }
      stopwatch[0].stop();

      if (print_timings) {
        std::cout << "thread 0: " << stopwatch[0].elapsed() << std::endl;
      }
    } else {

      for (int tid = 0; tid < num_threads; tid++) {
        threads.push_back(
          std::thread([&](uint64_t i0) {
            stopwatch[i0].start();
            for (uint64_t i = i0; i < ni; i += num_threads) {
              for (uint64_t j = 0; j < nj; j++) {
                for (uint64_t k = 0; k < nk; k++) {
                  f(i, j, k);
                }
              }
            }
            stopwatch[i0].stop();
          }, tid)
        );
      }

      for (int i = 0; i < num_threads; i++) {
        threads[i].join();
        if (print_timings) {
          std::cout << "thread " << i << ": " << stopwatch[i].elapsed() << std::endl;
        }
      }

      threads.clear();
    }
 }

  bool print_timings;
  int num_threads;
  std::vector< timer > stopwatch;
  std::vector< std::thread > threads;
};
