#include <Eigen/Dense>

using Eigen::MatrixXd;

MatrixXd load_image(const char *filename);
void write_matrix(const MatrixXd &mat, const char *filename);
