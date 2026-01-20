#pragma once

#include <cstdio>
#include <fstream>

#include "macros.hpp"
#include "iterator.hpp"

namespace femto {

template <MemorySpace memory, typename T>
void malloc(T*& ptr, size_t n) {
  if constexpr (memory == MemorySpace::CPU) {
    ptr = (T*)::malloc(n * sizeof(T));
  }
#ifdef __NVCC__
  if constexpr (memory == MemorySpace::GPU) {
    cudaMalloc(&ptr, n);
  }
  if constexpr (memory == MemorySpace::UNIFIED) {
    cudaMallocManaged(&ptr, n);
  }
#endif
}

template <MemorySpace memory, typename T>
void free(T* ptr) {
  if constexpr (memory == MemorySpace::CPU) {
    ::free(ptr);
  }
#ifdef __NVCC__
  if constexpr (memory == MemorySpace::GPU || memory == MemorySpace::UNIFIED) {
    cudaFree(ptr);
  }
#endif
}

template <MemorySpace src_memory, MemorySpace dst_memory, typename T>
auto copy(const T* begin, const T* end, T* output) {
  if constexpr (src_memory == MemorySpace::CPU &&
                dst_memory == MemorySpace::CPU) {
    std::copy(begin, end, output);
  }
#ifdef __NVCC__
  if constexpr (src_memory != MemorySpace::CPU ||
                dst_memory != MemorySpace::CPU) {
    cudaMemcpy(begin_, output_, (end - begin) * sizeof(T), cudaMemcpyDefault);
  }
#endif
}

#ifdef __NVCC__
template < typename T >
__global__ void fill_kernel(T * ptr, size_t n, T value) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < n) { 
    ptr[tid] = value;
  }
}
#endif

template <MemorySpace memory, typename T>
void fill_n(T* ptr, size_t n, const T& value) {

  if constexpr (memory == MemorySpace::CPU) {
    std::fill_n(ptr, n, value);
  }

#ifdef __NVCC__
  if constexpr (memory != MemorySpace::CPU) {
    int blocksize = 128;
    int gridsize = (blocksize + size_ - 1) / blocksize;
    fill_kernel<<<gridsize, blocksize>>>(begin_, size_, value);
    cudaDeviceSynchronize();
  }
#endif
}

}  // namespace femto

template <size_t d>
struct Indexable;

template <>
struct Indexable<1> {
  Indexable(size_t n0) : shape{n0} {}
  size_t index(size_t i) const { return i; };
  size_t shape;
};

template <>
struct Indexable<2> {
  Indexable(size_t n0, size_t n1) : shape{n0, n1} {}
  size_t index(size_t i, size_t j) const { return i * shape[1] + j; };
  size_t shape[2];
};

template <>
struct Indexable<3> {
  Indexable(size_t n0, size_t n1, size_t n2)
      : shape{n0, n1, n2}, strides{n1 * n2, n2} {}
  size_t index(size_t i, size_t j, size_t k) const {
    return i * strides[0] + j * strides[1] + k;
  };
  size_t shape[3];
  size_t strides[2];
};

template <class T, MemorySpace memory>
struct buffer {
  explicit buffer() {
    size_ = 0;
    ptr_ = nullptr;
  }

  explicit buffer(size_t n) { allocate(n); }

  buffer(const buffer<T, memory>& other) {
    allocate(other.size());
    femto::copy<memory, memory>(other.begin(), other.end(), begin());
  }

  buffer<T, memory>& operator=(const buffer<T, memory>& other) {
    allocate(other.size());
    femto::copy<memory, memory>(other.begin(), other.end(), begin());
    return *this;
  }

  template <MemorySpace other_memory>
  buffer(const buffer<T, other_memory>& other) {
    allocate(other.size());
    femto::copy<other_memory, memory>(other.begin(), other.end(), begin());
  }

  template <MemorySpace other_memory>
  buffer<T, memory>& operator=(const buffer<T, other_memory>& other) {
    resize(other.size());
    femto::copy<other_memory, memory>(other.begin(), other.end(), begin());
    return *this;
  }

  buffer(const std::vector<T>& other) {
    allocate(other.size());
    femto::copy<MemorySpace::CPU, memory>(other.begin(), other.end(), begin());
  }

  buffer<T, memory>& operator=(const std::vector<T>& other) {
    resize(other.size());
    femto::copy<MemorySpace::CPU, memory>(other.begin(), other.end(), begin());
    return *this;
  }

  ~buffer() noexcept { free(); }

  void resize(size_t new_size) {
    free();
    allocate(new_size);
  }

  operator T*() { return ptr_; }
  operator const T*() const { return ptr_; }

  const T* begin() const { return ptr_; }
  T* begin() { return ptr_; }

  const T* end() const { return ptr_ + size_; }
  T* end() { return ptr_ + size_; }

  size_t size() const { return size_; }

  void fill(const T& value) { femto::fill_n(ptr_, size_, value); }

  T& operator[](size_t i) { return ptr_[i]; }

  const T& operator[](size_t i) const { return ptr_[i]; }

  void write_to_file(std::string filename) {
    buffer<T, MemorySpace::CPU> host_copy(*this);

    std::ofstream outfile(filename, std::ios::out | std::ios::binary);
    outfile.write((char*)&size_, sizeof(size_t));
    outfile.write((char*)host_copy.begin(), sizeof(T) * size_);
    outfile.close();
  }

  void read_from_file(std::string filename) {
    std::ifstream infile;
    infile.open(filename, std::ios::in | std::ios::binary);
    infile.read((char*)&size_, sizeof(size_t));
    buffer<T, MemorySpace::CPU> host_copy(size_);
    infile.read((char*)host_copy.begin(), sizeof(T) * size_);
    infile.close();

    *this = host_copy;
  }

 private:
  // allocate memory in the specified memory space
  void allocate(size_t n) {
    size_ = n;
    femto::malloc<memory>(ptr_, n);
  }

  // free memory in the specified memory space
  void free() {
    if (ptr_ != NULL) {
      femto::free<memory>(ptr_);
      ptr_ = nullptr;
    }
  }

  T* ptr_;
  size_t size_;
};

template <class T, size_t dimension, MemorySpace memory = MemorySpace::CPU>
struct heap_array;

template <class T>
struct heap_array<T, 1, MemorySpace::CPU> : buffer<T, MemorySpace::CPU>, Indexable<1> {
  static constexpr auto memory = MemorySpace::CPU;
  explicit heap_array() : buffer<T, memory>{}, Indexable<1>{0} {}
  explicit heap_array(size_t n) : buffer<T, memory>{n}, Indexable<1>{n} {}
  T& operator[](size_t i) { return *this[i]; }
  T operator[](size_t i) const { return *this[i]; }
  T& operator()(size_t i) { return *this[i]; }
  T operator()(size_t i) const { return *this[i]; }
  size_t shape([[maybe_unused]] size_t i = 0) { return Indexable<1>::shape; }
};

template <class T>
struct heap_array<T, 2, MemorySpace::CPU> : public buffer<T, MemorySpace::CPU>, Indexable<2> {
  static constexpr auto memory = MemorySpace::CPU;
  explicit heap_array() : buffer<T, memory>{}, Indexable<2>{0, 0} {}
  explicit heap_array(size_t n0, size_t n1) : buffer<T, memory>{n0 * n1}, Indexable<2>{n0, n1} {}
  Iterator< T > operator()(size_t i) { 
    return Iterator<T>{&buffer<T, memory>::operator[](index(i,0)), &buffer<T, memory>::operator[](index(i+1,0))};
  }
  Iterator< const T > operator()(size_t i) const { 
    return Iterator<const T>{&buffer<T, memory>::operator[](index(i,0)), &buffer<T, memory>::operator[](index(i+1,0))};
  }
  T& operator()(size_t i, size_t j) { return buffer<T, memory>::operator[](index(i,j)); }
  T operator()(size_t i, size_t j) const { return buffer<T, memory>::operator[](index(i,j)); }
  size_t shape(size_t i) { return Indexable<2>::shape[i]; }
};

template <class T>
struct heap_array<T, 3, MemorySpace::CPU> : buffer<T, MemorySpace::CPU>, Indexable<3> {
  static constexpr auto memory = MemorySpace::CPU;
  explicit heap_array() : buffer<T, memory>{}, Indexable<3>{0, 0 ,0} {}
  explicit heap_array(size_t n0, size_t n1, size_t n2) : buffer<T, memory>{n0 * n1 * n2}, Indexable<3>{n0, n1, n2} {}
  T& operator()(size_t i, size_t j, size_t k) { return buffer<T, memory>::operator[](index(i,j,k)); }
  T operator()(size_t i, size_t j, size_t k) const { return buffer<T, memory>::operator[](index(i,j,k));}
  size_t shape(size_t i) { return Indexable<3>::shape[i]; }
};

template <class T, size_t dimension, MemorySpace memory>
struct view : Indexable< dimension >{

    view(const heap_array< T, dimension, memory > & arr) : Indexable<dimension>{arr} {

    }

    T * ptr_;

};
