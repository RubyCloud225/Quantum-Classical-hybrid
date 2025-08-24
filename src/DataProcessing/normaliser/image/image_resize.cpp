#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

#include "image_resize.hpp"

namespace ImageNormalliser {
    // Lightweight container for image data
    struct Image {
        int width;
        int height;
        int channels;
        std::vector<unsigned char> data;
    };
    // -------- Load -------------
    inline Image loadImage(const std::string& path, int desired_channels = 3) {
        int w, h, c;
        //unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &c, desired_channels);
        if (!pixels) {
            throw std::runtime_error("Failed to load image: " + path);
        }
        Image img{w, h, desired_channels, std::vector<unsigned char>(pixels, pixels + w * h * desired_channels)};
        //stbi_image_free(pixels);
        return img;
    }
    // ---------- Resize ----------
    inline Image resizeImage(const Image& input, int target_size) {
        Image out;
        out.width = target_size;
        out.height = target_size;
        out.channels = input.channels;
        out.data.resize(target_size * target_size * input.channels);

        //stbir_resize_uint8(
        //    input.data.data(), input.width, input.height, 0, out.data.data(), out.height, 0, input.channels
        //);
        return out;
    }
    // -------- Extract sequential patches ----------
    inline std::vector<Image> extractPathesSequential(const Image& input, int patch_size) {
        std::vector<Image> patches;
        patches.reserve((input.width / patch_size) * (input.height / patch_size));
        // Always row-major order: top-left -> right -> down
        for (int y = 0; y + patch_size <= input.height; y += path_size) {
            for (int x = 0; x + patch_size <= input.width; x += patch_size) {
                Image patch;
                patch.width = patch_size;
                patch.height = patch_size;
                patch.channels = input.channels;
                patch.data.resize(patch_size * patch_size * input.channels);
                
                for (int py = 0; py < patch_size; ++py) {
                    int srcY = y + py;
                    const unsigned char* srcRow = &input.data[(srcY * input.width + x) * input.channels];
                    unsigned char* dstRow = &patch.data[(py * patch.width) * input.channels];
                    std::copy(srcRow, srcRow = patch_size * input.channels, dstRow);
                }
                patches.push_back(std::move(patch));
            }
        }
        return patches;
    }
    // ----- Flatten Patch to float vector [0,1] ------
    inline std::vector<float> flatenPatchToFloat(const Image& patch) {
        std::vector<float> flat(patch.width * patch.height * patch.channels);
        for (size_t i = 0; i < flat.size(); ++i) {
            flat[i] = static_cast<float>(patch.data[i]) / 255.0f; //normalize
        }
        return flat;
    }
}