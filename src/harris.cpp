#include "harris.hh"

#include <iostream>

MatrixXd gauss_kernel(size_t ker_size)
{
    MatrixXd kernelX(2 * ker_size + 1, 2 * ker_size + 1);
    MatrixXd kernelY(2 * ker_size + 1, 2 * ker_size + 1);

    int size = ker_size;

    for (int i = -size; i <= size; i++)
    {
        for (int j = -size; j <= size; j++)
        {
            kernelY(i + size, j + size) = i;
            kernelX(i + size, j + size) = j;
        }
    }

    double sigma_sqr = (size * size) / 9.0;

    return (-(kernelX.array().square() + kernelY.array().square())
            / (2 * sigma_sqr))
        .exp();
}

std::pair<MatrixXd, MatrixXd> gauss_derivative_kernels(size_t ker_size)
{
    MatrixXd kernelX(2 * ker_size + 1, 2 * ker_size + 1);
    MatrixXd kernelY(2 * ker_size + 1, 2 * ker_size + 1);

    int size = ker_size;

    for (int i = -size; i <= size; i++)
    {
        for (int j = -size; j <= size; j++)
        {
            kernelY(i + size, j + size) = i;
            kernelX(i + size, j + size) = j;
        }
    }

    double sigma_sqr = (size * size) / 9.0;

    auto g = (-(kernelX.array().square() + kernelY.array().square())
              / (2 * sigma_sqr)).exp();

    MatrixXd gx = (-kernelX.array() / (sigma_sqr)) * g;
    MatrixXd gy = (-kernelY.array() / (sigma_sqr)) * g;

    return { gx, gy };
}

MatrixXd convolute(const MatrixXd &mat, const MatrixXd &ker)
{
    int sx = mat.cols(), sy = mat.rows();

    MatrixXd result(sy, sx);

    int w = ker.cols(), h = ker.rows();

    for (int y = 0; y < sy; y++)
    {
        for (int x = 0; x < sx; x++)
        {
            double convoluted_px = 0.;

            for (int k = 0; k < h; k++)
            {
                for (int j = 0; j < w; j++)
                {
                    int u = x - (j - w / 2), v = y - (k - w / 2);

                    if (u < 0 || u >= sx || v < 0 || v >= sy)
                        continue;

                    convoluted_px += ker.coeff(k, j) * mat.coeff(v, u);
                }
            }

            result(y, x) = convoluted_px;
        }
    }

    return result;
}

std::pair<MatrixXd, MatrixXd> gauss_derivatives(const MatrixXd &img,
                                                std::size_t ker_size)
{
    auto derivatives = gauss_derivative_kernels(ker_size);

    return { convolute(img, derivatives.first),
             convolute(img, derivatives.second) };
}

MatrixXd compute_harris_response(const MatrixXd &img,
                                 std::size_t derivative_ker_size,
                                 std::size_t opening_size)
{
    auto img_derivatives = gauss_derivatives(img, derivative_ker_size);

    auto &imx = img_derivatives.first;
    auto &imy = img_derivatives.second;

    auto gauss = gauss_kernel(opening_size);

    auto Wxx = convolute(imx.array().square(), gauss);
    auto Wxy = convolute(imx.array() * imy.array(), gauss);
    auto Wyy = convolute(imy.array().square(), gauss);

    auto Wdet = Wxx.array() * Wyy.array() - Wxy.array().square();
    auto Wtr = Wxx.array() + Wyy.array();

    return Wdet / (Wtr + 1.0);
}
