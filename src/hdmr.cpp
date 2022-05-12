#include <algorithm>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

#include "harris.hh"
#include "img_utils.hh"

void write_harris_resp(const MatrixXd &harris, const char *filename)
{
    auto max = harris.maxCoeff();
    auto scale_coeff = 255.0 / max;

    write_matrix(harris * scale_coeff, filename);
}

inline bool is_close(double a, double b, double rtol = 1e-5, double atol = 1e-8)
{
    return abs(a - b) <= (atol + rtol * abs(b));
}

std::vector<std::tuple<size_t, size_t, double>>
get_local_maximas(const MatrixXd &mat)
{
    auto width = mat.cols();
    auto height = mat.rows();

    std::vector<std::tuple<size_t, size_t, double>> local_maximums;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            auto block_i = i - (i > 0);
            auto block_j = j - (j > 0);
            auto block_h = 1 + (i > 0) + (i < height - 1);
            auto block_w = 1 + (j > 0) + (j < width - 1);

            double max =
                mat.block(block_i, block_j, block_h, block_w).maxCoeff();

            if (max > 0.0 && is_close(mat.coeff(i, j), max))
                local_maximums.emplace_back(i, j, max);
        }
    }

    return local_maximums;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
        return 1;

    auto mat_img = load_image(argv[1]);

    auto harris = compute_harris_response(mat_img);

    auto local_maximums = get_local_maximas(harris);

    std::sort(local_maximums.begin(), local_maximums.end(),
              [&](auto a, auto b) { return std::get<2>(a) > std::get<2>(b); });

    if (local_maximums.size() >= 2000)
        local_maximums.resize(2000);

    for (const auto &c : local_maximums)
        std::cout << std::get<0>(c) << "," << std::get<1>(c) << ","
                  << std::get<2>(c) << "\n";

    return 0;
}
