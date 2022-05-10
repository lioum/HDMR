#include <Eigen/Dense>
#include <cstddef>
#include <utility>

using Eigen::MatrixXd;

MatrixXd compute_harris_response(const MatrixXd &img,
                                 std::size_t derivative_ker_size = 3,
                                 std::size_t opening_size = 3);
