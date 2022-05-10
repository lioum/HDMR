#include <iostream>
#include "img_utils.hh"
#include "gauss.hh"

int main (int argc, char *argv[])
{
    if (argc < 3)
        return 1;

    auto mat_img = load_image(argv[1]);


    write_matrix(mat_img, argv[2]);

    return 0;
}
