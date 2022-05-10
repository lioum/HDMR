#include <Eigen/Dense>
#include <cstddef>
#include <utility>

using Eigen::MatrixXd;

MatrixXd gauss_kernel(size_t ker_size);
std::pair<MatrixXd, MatrixXd> gauss_derivatives(MatrixXd &img, std::size_t ker_size);
