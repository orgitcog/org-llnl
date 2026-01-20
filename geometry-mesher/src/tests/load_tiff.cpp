#if UM_TIFF_SUPPORT

#include "timer.hpp"
#include "geometry/image.hpp"

#include <iostream>

using namespace geometry;

int main(int argc, char *argv[]) {
  if (argc == 1) {
    std::cout << "error: must provide a tiff file to read" << std::endl;
    exit(1);
  }

  timer stopwatch;
  
  stopwatch.start();
  Image im = import_tiff(argv[1]);
  stopwatch.stop();
  std::cout << stopwatch.elapsed() << std::endl;

  std::cout << im(799, 21) << std::endl;
  std::cout << im(799, 22) << std::endl;
  std::cout << im.interpolate(799.0f, 21.0f) << std::endl;
  std::cout << im.interpolate(799.0f, 21.5f) << std::endl;
  std::cout << im.interpolate(799.0f, 22.0f) << std::endl;
}

#else
int main() { return 0; }
#endif