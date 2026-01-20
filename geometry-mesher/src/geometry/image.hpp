#include <vector>
#include <string>
#include <cmath>
#include <cstdint>

namespace geometry {

struct Image {

    uint32_t width, height;
    std::vector< float > pixel_values;

    float & operator()(uint32_t row, uint32_t col) {
        return pixel_values[row * width + col];
    };
    const float & operator()(uint32_t row, uint32_t col) const {
        return pixel_values[row * width + col];
    };

    float interpolate(float row, float col) {
        uint32_t i = uint32_t(floor(row));
        uint32_t j = uint32_t(floor(col));

        float s = row - i;
        float t = col - j;
        
        return (1.0f - s) * (1.0f - t) * (*this)(i  , j  ) + 
                       s  * (1.0f - t) * (*this)(i+1, j  ) + 
               (1.0f - s) *         t  * (*this)(i  , j+1) + 
                       s  *         t  * (*this)(i+1, j+1);
    };

};

Image import_tiff(std::string filename);

}