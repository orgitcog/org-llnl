#include "geometry/parse_dat.cpp"

using namespace geometry;

int main() {

  constexpr int x = 0;
  constexpr int y = 1;
  constexpr int z = 2;

  std::vector < ColumnInfo > columns = {
    {"ProgPosFbk#00[0]", x},
    {"ProgPosFbk#01[0]", y},
    {"ProgPosFbk#02[0]", z},
    {"Ain_0#00[0]"},
    {"Ain_1#00[0]"}
  }; 

  auto raw_data = combine< std::vector< float > >({
    parse_datfile("../data/Exp3_Segment1.dat", columns),
    parse_datfile("../data/Exp3_Segment2.dat", columns)
  });

}