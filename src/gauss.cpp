#include "gauss.hh"

using Eigen::exp;

MatrixXd gauss_kernel(size_t ker_size)
{
    int size = (ker_size % 2 == 0) ? ker_size + 1 : ker_size;

    MatrixXd kernelX(ker_size, ker_size);
    MatrixXd kernelY(ker_size, ker_size);

    int offset = size / 2;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            kernelY(i,j) = i - offset;
            kernelX(i,j) = j - offset;
        }
    }

    double sigma_sqr = (ker_size * ker_size) / 9.0;

    return exp(-(kernelX.array().square() * kernelY.array().square()) / sigma_sqr).matrix();
}
std::pair<MatrixXd, MatrixXd> gauss_derivative_kernels(size_t i);

std::pair<MatrixXd, MatrixXd> gauss_derivatives(MatrixXd &img,
                                                std::size_t ker_size);
