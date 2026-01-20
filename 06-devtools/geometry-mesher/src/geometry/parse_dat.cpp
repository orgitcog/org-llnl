#include "parse_dat.hpp"

#include <string>
#include <fstream>
#include <iostream>

namespace geometry {

enum class ParseState { HEADER, DATA, FOOTER };

std::vector< float > to_float(std::vector< std::string > strings) {
  std::vector< float > output(strings.size());
  for (std::size_t i = 0; i < strings.size(); i++) {
    output[i] = std::stof(strings[i]);
  }
  return output;
}

std::vector< std::string > split_string(const std::string & str, std::string delimiter) {
  size_t pos_start = 0;
  size_t pos_end = 0;
  size_t delim_len = delimiter.length();

  std::vector<std::string> entries;
  while ((pos_end = str.find (delimiter, pos_start)) != std::string::npos) {
    entries.push_back(str.substr(pos_start, pos_end - pos_start));
    pos_start = pos_end + delim_len;
  }

  return entries;
}

std::vector< float > parse_floats_separated_by(const std::string & str, std::string delimiter) {
  size_t pos_start = 0;
  size_t pos_end = 0;
  size_t delim_len = delimiter.length();

  std::vector<float> entries;
  while ((pos_end = str.find (delimiter, pos_start)) != std::string::npos) {
    float value = std::stof(str.substr(pos_start, pos_end - pos_start));
    entries.push_back(value);
    pos_start = pos_end + delim_len;
  }

  return entries;
}

std::vector< Capsule > parse_datfile(std::string filename, std::array< int, 3 > columns, float r) {

  std::ifstream infile(filename);

  if (!infile) { 
    std::cout << "invalid file: " << filename << ", exiting..." << std::endl;
    exit(1);
  }

  std::vector< vec3f > points;
  std::vector<float> scale;
  ParseState state = ParseState::HEADER;

  std::string line;

  auto line_starts_with = [&](std::string str) {
    return (line.substr(0, str.size()) == str);
  };

  while (std::getline(infile, line)){
    if (state == ParseState::HEADER) {
      if (line[0] != ';') {
        state = ParseState::DATA;
      }
    }

    if (state == ParseState::DATA) {
      if (line[0] != ';') {
        std::vector< float > entries = parse_floats_separated_by(line, " ");
        points.push_back({entries[columns[0]], entries[columns[1]], entries[columns[2]]});
      } else {
        state = ParseState::FOOTER;
      }
    }

    if (state == ParseState::FOOTER) {
      if (line_starts_with("; CountsPerUnit=")) {
        scale.push_back(1.0f / std::stof(line.substr(16)));
      }
    }

  }

  for (auto & point : points) {
    point[0] *= scale[0];
    point[1] *= scale[1];
    point[2] *= scale[2];
  }

  std::vector< Capsule > capsules(points.size() - 1);

  for (uint32_t i = 0; i < points.size() - 1; i++) {
    capsules[i] = Capsule{points[i], points[i+1], r, r};
  }

  return capsules;

}

int index_of(std::string needle, std::vector< std::string > haystack) {
  int index = 0;
  for (auto item : haystack) {
    if (needle == item) {
      return index;
    } else {
      index++;
    }
  }

  std::cout << "Error: could not find \"" << needle << "\" among the list entries:\n";
  for (auto name : haystack) {
    std::cout << "  " << name << std::endl;
  }
  std::cout << "exiting..." << std::endl;
  exit(1);
}


std::vector< std::vector< float > > parse_datfile(std::string filename, std::vector< ColumnInfo > columns) {

  std::ifstream infile(filename);

  if (!infile) { 
    std::cout << "invalid file: " << filename << ", exiting..." << std::endl;
    exit(1);
  } else {
    std::cout << "parsing file: " << filename << " ..." << std::endl;
  }

  std::vector< std::vector < float > > output;
  std::vector < int > column_indices;

  std::vector<float> scale;

  int line_number = 0;
  ParseState state = ParseState::HEADER;

  std::string line;

  auto line_starts_with = [&](std::string str) {
    return (line.substr(0, str.size()) == str);
  };

  while (std::getline(infile, line)) {
    if (state == ParseState::HEADER) {
      if (line_number == 3) {
        auto header_names = split_string(line.substr(2), " ");
        for (auto [column_name, unused] : columns) {
          column_indices.push_back(index_of(column_name, header_names));
          std::cout << column_name << ": " << column_indices.back() << std::endl;
        }
      }

      if (line[0] != ';') {
        state = ParseState::DATA;
      }
    }

    if (state == ParseState::DATA) {
      if (line[0] != ';') {
        std::vector< float > entries = parse_floats_separated_by(line, " ");
        std::vector< float > row(columns.size());
        for (std::size_t i = 0; i < columns.size(); i++) {
          row[i] = entries[column_indices[i]];
        }
        output.push_back(row);
      } else {
        state = ParseState::FOOTER;
      }
    }

    if (state == ParseState::FOOTER) {
      if (line_starts_with("; CountsPerUnit=")) {
        scale.push_back(1.0f / std::stof(line.substr(16)));
      }
    }

    line_number++;
  }

  for (auto & row : output) {
    for (std::size_t i = 0; i < columns.size(); i++) {
      if (columns[i].component != -1) {
        row[i] *= scale[columns[i].component];
      }
    }
  }

  return output;

}

}