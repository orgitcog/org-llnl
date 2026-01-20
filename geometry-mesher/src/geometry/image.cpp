#include "image.hpp"

#include <tiffio.h>

#include <iostream>

namespace geometry {

Image import_tiff(std::string filename) {

  Image im{};

  TIFF *tif = TIFFOpen(filename.c_str(), "r");

  if (!tif) {
    std::cout << "unable to open file: `" << filename << "`" << std::endl;
    exit(1);
  }

  uint16_t photometric;
  uint16_t bits_per_sample;

  bool something_went_wrong = false;

#ifndef VERBOSE
  TIFFSetErrorHandler(nullptr);
  TIFFSetWarningHandler(nullptr);
#endif

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &im.width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &im.height);

  TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric);
  if (photometric != 0 && photometric != 1) {
    std::cout << "unsupported photometric value, only grayscale images are supported" << std::endl;
    something_went_wrong = true;
  }

  TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
  if (bits_per_sample != 8 && bits_per_sample != 16) {
    std::cout << "unsupported bits per sample, only 8-bit and 16-bit values are " "supported" << std::endl;
    something_went_wrong = true;
  }

  if (something_went_wrong) {
    std::cout << "encountered errors when importing tiff file, exiting..." << std::endl;
    exit(1);
  }

  [[maybe_unused]] bool white_is_zero = (photometric == 0);
  float scale = 1.0f / float(uint32_t(1) << (bits_per_sample - 1));

  uint32_t num_pixels = im.width * im.height;
  im.pixel_values.resize(num_pixels);

  std::vector< uint16_t > buf(im.width);
  for (uint32_t row = 0; row < im.height; row++) {
    TIFFReadScanline(tif, &buf[0], row);
    for (uint32_t col = 0; col < im.width; col++) {
      im(row, col) = float(buf[col]) * scale;
    }
  }

  return im;
}

}