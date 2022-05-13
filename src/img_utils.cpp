#include "img_utils.hh"

#include <algorithm>
#include <iostream>

#include "stb_image.h"
#include "stb_image_write.h"

MatrixXd load_image(const char *filename)
{
    int width, height, bpp;

    auto bytes = stbi_load(filename, &width, &height, &bpp, 3);

    auto mat = MatrixXd(height, width);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            double r = bytes[i * 3 * width + 3 * j];
            double g = bytes[i * 3 * width + 3 * j + 1];
            double b = bytes[i * 3 * width + 3 * j + 2];

            mat(i, j) = 0.299 * r + 0.587 * g + 0.114 * b;
        }
    }

    stbi_image_free(bytes);

    return mat;
}

void write_matrix(const MatrixXd &mat, const char *filename)
{
    int w = mat.cols(), h = mat.rows();

    auto bytes = new uint8_t[3 * w * h];

    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            bytes[i * 3 * w + j * 3] = mat.coeff(i, j);
            bytes[i * 3 * w + j * 3 + 1] = mat.coeff(i, j);
            bytes[i * 3 * w + j * 3 + 2] = mat.coeff(i, j);
        }
    }

    stbi_write_png(filename, w, h, 3, bytes, 3 * w);

    delete[] bytes;
}
