// Copyright 2019 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

#include <krowkee/util/compact_iterable_map.hpp>

#include <krowkee/hash/util.hpp>

#include <krowkee/util/tests.hpp>

#include <boost/container/flat_map.hpp>

#include <algorithm>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

typedef krowkee::util::compact_iterable_map<int, int> cim_t;
typedef std::map<int, int> map_t;
typedef boost::container::flat_map<int, int> bm_t;
typedef std::pair<int, int> pair_t;

typedef std::chrono::system_clock Clock;
typedef std::chrono::nanoseconds ns_t;

std::ostream &operator<<(std::ostream &os, const pair_t &pair) {
  os << "(" << pair.first << "," << pair.second << ")";
  return os;
}

std::vector<int> get_random_vector(const int count, const std::uint32_t seed) {
  std::vector<int> vec;
  vec.reserve(count);
  for (int i(1); i <= count; ++i) { vec.push_back(i); }
  std::random_device rd;
  std::mt19937 g(seed);
  std::shuffle(std::begin(vec), std::end(vec), g);
  return vec;
}

template <typename MapType>
MapType make_map(const int threshold = 1000) {
  if constexpr (std::is_same_v<MapType, cim_t>) {
    return cim_t(threshold);
    // } else if constexpr (std::is_same_v<MapType, cbm_t>) {
    //   return cbm_t(threshold);
  } else {
    return MapType();
  }
}

template <typename MapType>
auto clock_map(const std::vector<int> to_insert, const int count,
               const int threshold = 1000) {
  MapType map(make_map<MapType>(threshold));
  auto start(Clock::now());
  std::for_each(std::begin(to_insert), std::end(to_insert), [&](const int i) {
    auto [iter, success] = map.insert(pair_t(i, i));
  });
  auto end(Clock::now());
  return std::chrono::duration_cast<ns_t>(end - start).count();
}

void benchmark_insert(const std::vector<int> counts,
                      const std::vector<double> threshes,
                      const std::uint32_t seed, const bool verbose) {
  if (verbose == false) {
    std::cout << "count, std::map, boost::container::flat_map";
    for (const double thresh_ratio : threshes) {
      std::cout << ", " << thresh_ratio;
    }
    std::cout << std::endl;
  }
  for (const int count : counts) {
    std::vector<int> to_insert = get_random_vector(count, seed);
    auto map_ns(clock_map<map_t>(to_insert, count));
    // std::cout << count << ", " << map_ns << ", NaN";
    // for (const double thresh_ratio : threshes) {
    //   const int threshold(count * thresh_ratio);
    //   auto cim_ns(clock_map<cim_t>(to_insert, count, threshold));
    //   std::cout << ", " << cim_ns;
    // }
    // std::cout << std::endl;
    // return;
    auto boost_ns(clock_map<bm_t>(to_insert, count));
    if (verbose == true) {
      std::cout << "On inserting " << count << " elements:" << std::endl;
      std::cout << "\tstd::map takes " << map_ns << " ns" << std::endl;
      std::cout << "\tboost::container::flat_map takes " << boost_ns << " ns"
                << std::endl;
    } else {
      std::cout << count << ", " << map_ns << ", " << boost_ns;
    }
    for (const double thresh_ratio : threshes) {
      const int threshold(count * thresh_ratio);
      auto cim_ns(clock_map<cim_t>(to_insert, count, threshold));

      if (verbose == true) {
        std::cout << "\t(std::map) thresh = " << threshold << " takes "
                  << cim_ns << " ns (" << ((double)cim_ns / (double)map_ns)
                  << " slowdown vs std::map) and ("
                  << ((double)cim_ns / (double)boost_ns)
                  << " slowdown vs boost::container::flat_map)" << std::endl;
      } else {
        std::cout << ", " << cim_ns;
      }
    }
    std::cout << std::endl;
  }
}

int main(int argc, char **argv) {
  std::uint32_t seed(0);
  bool verbose(false);

  std::vector<int> counts{1000, 10000, 100000, 1000000};
  std::vector<double> threshes{0.001, 0.005, 0.01, 0.1};

  int argi(1);
  while (argi < argc) {
    const char *arg = argv[argi];
    if (std::strcmp(arg, "-v") == 0) {
      verbose = true;
    } else if (std::strcmp(arg, "--seed") == 0) {
      seed = std::atol(argv[++argi]);
    } else {
      break;
    }
    ++argi;
  }
  if ((argc - argi) != 0) {
    std::cout << "usage:  " << argv[0] << " [-v | --seed SEED]" << std::endl;
    exit(-1);
  }

  benchmark_insert(counts, threshes, seed, verbose);
  return 0;
}
