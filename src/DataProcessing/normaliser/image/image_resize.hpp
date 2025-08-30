#ifndef IMAGE_RESIZE_HPP
#define IMAGE_RESIZE_HPP
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

namespace ImageNormaliser {
    //lightweight container for image data
    struct Image {
        int width;
        int height;
        int channels;
        std::vector<unsigned char> data; // pixel buffer [H * W * C]
    };
    //-------- Load-----------------
    inline Image loadImage(const std::string& path, int desired_channels = 3, bool flip_vertically = false, unsigned char* pixels = nullptr);
    //---------Resize --------------
    inline Image resizeImage(const Image& input, int target_size);
    // --------- Extract Sequential Patches ---------
    inline std::vector<Image> extractPathesSequential(const Image& input, int patch_size);
    // ----------- Flatten Patch to Float Vector [0,1] -------
    inline std::vector<float> flattenPathToFloat(const Image& patch);
    // ----------- Free Image Pixels -------------
    unsigned char* pixels;
    int patch_size;

}
#endif //image_resize