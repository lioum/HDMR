#include <iostream>

#include "harris.hh"
#include "img_utils.hh"

void write_harris_resp(const MatrixXd &harris, const char *filename) {

    auto max = harris.maxCoeff();
    auto scale_coeff = 255.0 / max;

    write_matrix(harris * scale_coeff, filename);
}

int main(int argc, char *argv[])
{
    if (argc < 3)
        return 1;

    auto mat_img = load_image(argv[1]);

    auto harris = compute_harris_response(mat_img);

    write_harris_resp(harris, argv[2]);

    return 0;
}
