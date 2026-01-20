// Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
// krowkee Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <krowkee/util/compact_iterable_map.hpp>

#include <krowkee/hash/util.hpp>

#include <krowkee/util/tests.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

typedef krowkee::util::compact_iterable_map<int, int> cim_t;
typedef std::map<int, int>                            map_t;
typedef std::pair<int, int>                           pair_t;

typedef std::chrono::system_clock Clock;
typedef std::chrono::nanoseconds  ns_t;

std::ostream &operator<<(std::ostream &os, const pair_t &pair) {
  os << "(" << pair.first << "," << pair.second << ")";
  return os;
}

std::vector<int> get_random_vector(const int count, const std::uint32_t seed) {
  std::vector<int> vec;
  vec.reserve(count);
  for (int i(1); i <= count; ++i) {
    vec.push_back(i);
  }
  std::random_device rd;
  std::mt19937       g(seed);
  std::shuffle(std::begin(vec), std::end(vec), g);
  return vec;
}

struct insert_check {
  const char *name() { return "insert check"; }

  void operator()(cim_t &cm, const std::vector<int> to_insert, const int count,
                  const bool verbose) const {
    int  counter(0);
    bool all_success(true);

    auto cm_start(Clock::now());
    std::for_each(std::begin(to_insert), std::end(to_insert), [&](const int i) {
      auto [iter, success] = cm.insert({i, i});
      counter++;
      if (success == false) {
        std::cout << "FAILED " << pair_t{i, i} << std::endl;
        all_success == false;
      }
      if (verbose == true) {
        std::cout << "on " << counter << "th insert (size " << cm.size()
                  << ") : " << pair_t{i, i} << ":" << std::endl;
        std::cout << cm.print_state() << std::endl << std::endl;
      }
    });
    std::cout << ((all_success == true) ? "passed" : "failed")
              << " iterative insert test" << std::endl;

    map_t map;
    auto  map_start(Clock::now());
    std::for_each(std::begin(to_insert), std::end(to_insert), [&](const int i) {
      auto [iter, success] = map.insert({i, i});
    });
    auto end(Clock::now());

    {  // Test whether attempting to insert a dynamic element correctly fails
      auto [iter, success] = cm.insert({to_insert[count - 1], 1});
      std::cout << ((success == false) ? "passed" : "failed")
                << " dynamic insert test" << std::endl;
    }
    {  // Test whether attempting to insert an archive element correctly fails
      auto [iter, success] = cm.insert({to_insert[0], 1});
      std::cout << ((success == false) ? "passed" : "failed")
                << " archive insert test" << std::endl;
    }
    auto cm_ns(std::chrono::duration_cast<ns_t>(map_start - cm_start).count());
    auto map_ns(std::chrono::duration_cast<ns_t>(end - map_start).count());
    std::cout << "Inserted " << count
              << " elements into compact map with compaction threshold "
              << cm.threshold() << " in " << cm_ns << " ns versus std::map in "
              << map_ns << " ns (" << ((double)cm_ns / (double)map_ns)
              << " slowdown)" << std::endl;
  }
};

struct find_check {
  const char *name() { return "find check"; }

  void operator()(cim_t &cm, const std::vector<int> &to_insert, const int count,
                  const bool verbose) const {
    {  // Test whether * returns the correct reference from the find() iterator
      auto q = cm.find(to_insert[count - 1]);
      std::cout << (((*q).second == to_insert[count - 1]) ? "passed" : "failed")
                << " positive find ((*).second) test" << std::endl;
    }
    {  // Test whether -> returns the correct reference from the find() iterator
      auto q = cm.find(to_insert[count - 1]);
      std::cout << ((q->second == to_insert[count - 1]) ? "passed" : "failed")
                << " positive find (*->second) test" << std::endl;
    }
    {  // Test whether find() returns end() iterator on unseen element.
      auto q = cm.find(-10);
      std::cout << ((q == std::end(cm)) ? "passed" : "failed")
                << " negative find test" << std::endl;
    }
  }
};

struct iterator_check {
  const char *name() { return "iterator check"; }

  void operator()(cim_t &cm, const std::uint32_t count,
                  const bool verbose) const {
    std::cout << "state: " << std::endl;
    std::cout << cm.print_state() << std::endl << std::endl;
    std::cout << "conventional ascenting loop: " << std::endl;
    for (const auto &q : cm) {
      std::cout << q << " ";
    }
    std::cout << std::endl << std::endl;

    auto iter(std::begin(cm));
    std::cout << "ascending: " << std::endl;
    for (; iter != std::end(cm); ++iter) {
      std::cout << *iter << " ";
    }
    std::cout << std::endl << std::endl;

    // std::cout << "descending: " << std::endl;
    // --iter;
    // for (; iter != std::begin(cm); --iter) { std::cout << *iter << " "; }
    // std::cout << *iter << std::endl;

    std::cout << iter << std::endl;
    std::cout << std::end(cm) << std::endl;

    std::cout << "descending: " << std::endl;
    // --iter;
    for (int i(0); i < count; ++i) {
      --iter;
      std::cout << *iter << "; ";
    }
    std::cout << *iter << std::endl;
  }
};

int main(int argc, char **argv) {
  // std::uint32_t count(23);
  std::uint32_t seed(0);
  std::size_t   thresh(5);
  bool          verbose(false);

  int argi(1);
  while (argi < argc) {
    const char *arg = argv[argi];
    if (std::strcmp(arg, "-v") == 0) {
      verbose = true;
    } else if (std::strcmp(arg, "--seed") == 0) {
      seed = std::atol(argv[++argi]);
    } else if (std::strcmp(arg, "--thresh") == 0) {
      thresh = std::atoll(argv[++argi]);
    } else {
      break;
    }
    ++argi;
  }
  if ((argc - argi) != 1) {
    std::cout << "usage:  " << argv[0]
              << " [-v | --seed SEED | --thresh THRESH] <count>" << std::endl;
    exit(-1);
  }

  const std::uint32_t count(std::atol(argv[argi++]));

  std::vector<int> to_insert = get_random_vector(count, seed);
  cim_t            cm(thresh);

  do_test<insert_check>(cm, to_insert, count, verbose);
  do_test<find_check>(cm, to_insert, count, verbose);
  do_test<iterator_check>(cm, count, verbose);
  return 0;
}
