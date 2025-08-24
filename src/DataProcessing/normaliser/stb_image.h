#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// load the image 

int w, h, c;
unsigned char* pixels = stbi_load("dog.jpg", &w, &h, &c, 3); // force3 channels (RBG)

if(!pixels) {
    throw std::runtime_error("Failed to load image");
}
// use pixels [w*h*3 bytes]

// when done:
stbi_image_free(pixels);