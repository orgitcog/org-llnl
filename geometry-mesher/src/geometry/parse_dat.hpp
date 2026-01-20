#pragma once

#include "geometry/geometry.hpp"

namespace geometry {

std::vector< Capsule > parse_datfile(std::string filename, std::array< int, 3 > columns, float r);

struct ColumnInfo{
  ColumnInfo(std::string str, int c = -1) : name(str), component(c) {}
  std::string name;
  int component;
};

std::vector< std::vector< float > > parse_datfile(std::string filename, std::vector< ColumnInfo > columns);

}