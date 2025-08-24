#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

std::vector<unsigned char> out(224*224*3); // target size
stbir_resize_uint8(
    pixels, w, h, 0, // src buffer + src stride
    out.data(), 224, 224, 0, // dst buffer + dst stride
    3 // channels
);