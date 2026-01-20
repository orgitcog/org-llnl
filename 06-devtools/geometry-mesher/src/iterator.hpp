#pragma once

template < typename T >
struct Iterator {
  T * begin() { return begin_; };
  T * end() { return end_; };
  T * begin_;
  T * end_;

  T & operator[](size_t i) { return begin_[i]; }
  const T & operator[](size_t i) const { return begin_[i]; }

  template < typename indexable >
  void operator+=(const indexable & f) {
    size_t n = end_ - begin_;
    for (size_t i = 0; i < n; i++) {
      begin_[i] += f(i);
    }; 
  }
};
