#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template <typename T>
std::vector<T> read_binary(std::string filename) {
  std::vector<T> buffer;

  std::ifstream infile(filename, std::ios::binary);

  if (infile) {
    // std::cout << "file found: " << filename << std::endl;
    infile.seekg(0, std::ios::end);
    std::streampos filesize = infile.tellg();
    infile.seekg(0, std::ios::beg);

    if (filesize % sizeof(T) != 0) {
      std::cout << "file " << filename << " does not define an integer number of data, each of expected size "
                << sizeof(T) << " bytes" << std::endl;
      exit(1);
    }

    std::size_t num_entries = filesize / sizeof(T); 
    buffer = std::vector<T>(num_entries);
    infile.read((char*)&buffer[0], filesize);
  } else {
    std::cout << "file not found: " << filename << std::endl;
  }

  infile.close();

  return buffer;
}

template <typename T>
void write_binary(const std::vector<T>& buffer, std::string filename) {
  std::ofstream outfile(filename, std::ios::binary);

  if (outfile) {
    outfile.write((char*)&buffer[0], sizeof(T) * buffer.size());
  } else {
    std::cout << "file not found: " << filename << std::endl;
  }

  outfile.close();
}
