#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void grayscale(unsigned char *src, float *dst, int height, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float r = src[y * 3 * width + 3 * x];
    float g = src[y * 3 * width + 3 * x + 1];
    float b = src[y * 3 * width + 3 * x + 2];

    dst[y * width + x] = 0.299 * r + 0.587 * g + 0.114 * b;
}

__global__ void multiply_ew(float *lhs, float *rhs, float *dst, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    dst[i] = lhs[i] * rhs[i];
}

__device__ float gauss_coeff(int ker_x, int ker_y, int ker_radius, float sigma_sqr)
{
    return expf(-((ker_x * ker_x + ker_y * ker_y) / (2 * sigma_sqr)));
}

__global__ void gauss_kernel_init(float *ker, int ker_size, float sigma_sqr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= ker_size || y >= ker_size)
        return;

    int ker_radius = ker_size / 2;
    int ker_x = x - ker_radius;
    int ker_y = y - ker_radius;

    ker[y * ker_size + x] = gauss_coeff(ker_x, ker_y, ker_radius, sigma_sqr);
}

__global__ void gauss_kernel_derivatives_init(float *gx, float *gy, int ker_size, float sigma_sqr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= ker_size || y >= ker_size)
        return;

    int ker_radius = ker_size / 2;
    int ker_x = x - ker_radius;
    int ker_y = y - ker_radius;

    gx[y * ker_size + x] =
        gauss_coeff(ker_x, ker_y, ker_radius, sigma_sqr) * (-ker_x / sigma_sqr);
    gy[y * ker_size + x] =
        gauss_coeff(ker_x, ker_y, ker_radius, sigma_sqr) * (-ker_y / sigma_sqr);
}

__global__ void convolve2D(float *src, float *kernel, float *dst, int height, int width, int mask_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float Pvalue = 0;
    int half_size = mask_size / 2;

    for (int k = 0; k < mask_size; k++)
    {
        for (int j = 0; j < mask_size; j++)
        {
            int u = x - (j - half_size), v = y - (k - half_size);

            if (u < 0 || u >= width || v < 0 || v >= height)
                continue;
            Pvalue += src[v * width + u] * kernel[k * mask_size + j];
        }
    }

    dst[y * width + x] = Pvalue;
}

__global__ void compute_response(float *Wxx, float *Wyy, float *Wxy, float *dst, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    float Wdet = Wxx[i] * Wyy[i] - Wxy[i] * Wxy[i];
    float Wtr = Wxx[i] + Wyy[i];

    dst[i] = Wdet / (Wtr + 1);
}

void compute_harris_response(const unsigned char *img_rgb, float *harris_response, int width,
                          int height, size_t derivative_ker_size,
                          size_t opening_size)
{
    size_t size = width * height;
    
    unsigned char *d_img_rgb; 
    cudaMalloc(&d_img_rgb, 3 * size * sizeof(unsigned char));
    float *d_img_gray;        
    cudaMalloc(&d_img_gray, size * sizeof(float));

    cudaMemcpy(d_img_rgb, img_rgb, 3 * size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threads(32,32);
    dim3 blocks((width+threads.x-1)/threads.x,
                (height+threads.y-1)/threads.y);   

    grayscale<<<blocks, threads>>>(d_img_rgb, d_img_gray, height, width);
    cudaDeviceSynchronize();

    float *d_g, *d_gx, *d_gy, *d_imx, *d_imy, *d_Wxx, *d_Wyy, *d_Wxy, *d_response;
    size_t x_opening_size = 2 * opening_size + 1; 
    size_t x_derivative_ker_size = 2 * derivative_ker_size + 1;

    cudaMalloc(&d_g, x_opening_size * x_opening_size * sizeof(float));
    cudaMalloc(&d_gx, x_derivative_ker_size * x_derivative_ker_size * sizeof(float));
    cudaMalloc(&d_gy, x_derivative_ker_size * x_derivative_ker_size * sizeof(float));

    cudaMalloc(&d_imx, size * sizeof(float));
    cudaMalloc(&d_imy, size * sizeof(float));

    cudaMalloc(&d_Wxx, size * sizeof(float));
    cudaMalloc(&d_Wyy, size * sizeof(float));
    cudaMalloc(&d_Wxy, size * sizeof(float));

    cudaMalloc(&d_response, size * sizeof(float));

    dim3 blocks_opening((x_opening_size + threads.x - 1) / threads.x,
                        (x_opening_size + threads.y - 1) / threads.y);
    dim3 blocks_derivative_ker(
        ((x_derivative_ker_size + threads.x - 1) / threads.x,
         (x_derivative_ker_size + threads.y - 1) / threads.y));

    float opening_sigma_sqr = (opening_size * opening_size) / 9.0;
    float derivative_sigma_sqr =
        (derivative_ker_size * derivative_ker_size) / 9.0;

    gauss_kernel_init<<<blocks_opening, threads>>>(d_g, x_opening_size,
                                                   opening_sigma_sqr);
    gauss_kernel_derivatives_init<<<blocks_derivative_ker, threads>>>(
        d_gx, d_gy, x_derivative_ker_size, derivative_sigma_sqr);

    cudaDeviceSynchronize();

    convolve2D<<<blocks,threads>>>(d_img_gray, d_gx, d_imx, height, width, x_derivative_ker_size);
    convolve2D<<<blocks,threads>>>(d_img_gray, d_gy, d_imy, height, width, x_derivative_ker_size);

    cudaDeviceSynchronize();

    float *d_imx2, *d_imy2, *d_imxy;
    cudaMalloc(&d_imx2, size * sizeof(float));
    cudaMalloc(&d_imy2, size * sizeof(float));
    cudaMalloc(&d_imxy, size * sizeof(float));

    int nb_threads = threads.x * threads.y;

    multiply_ew<<<(nb_threads + size - 1) / nb_threads, nb_threads>>>(d_imx, d_imx, d_imx2, size);
    multiply_ew<<<(nb_threads + size - 1) / nb_threads, nb_threads>>>(d_imy, d_imy, d_imy2, size);
    multiply_ew<<<(nb_threads + size - 1) / nb_threads, nb_threads>>>(d_imx, d_imy, d_imxy, size);

    cudaDeviceSynchronize();

    cudaFree(d_imx);
    cudaFree(d_imy);

    convolve2D<<<blocks,threads>>>(d_imx2, d_g, d_Wxx, height, width, x_derivative_ker_size);
    convolve2D<<<blocks,threads>>>(d_imy2, d_g, d_Wyy, height, width, x_derivative_ker_size);
    convolve2D<<<blocks,threads>>>(d_imxy, d_g, d_Wxy, height, width, x_derivative_ker_size);

    cudaDeviceSynchronize();

    compute_response<<<(nb_threads + size - 1) / nb_threads, nb_threads>>>(
        d_Wxx, d_Wyy, d_Wxy, d_response, size);

    cudaDeviceSynchronize();

    cudaMemcpy(harris_response, d_response, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_img_gray);
    cudaFree(d_img_rgb);

    cudaFree(d_g);
    cudaFree(d_gx);
    cudaFree(d_gy);

    cudaFree(d_imx2);
    cudaFree(d_imy2);
    cudaFree(d_imxy);
    
    cudaFree(d_Wxx);
    cudaFree(d_Wyy);
    cudaFree(d_Wxy);
    
    cudaFree(d_response);
}

int main(int argc, char *argv[])
{
    if (argc <= 1)
        return 2;

    int width, height, bpp;
    auto img = stbi_load(argv[1], &width, &height, &bpp, 3);
    
    float *harris_response = new float[width * height];

    compute_harris_response(img, harris_response, width, height, 3, 3);

    stbi_image_free(img);

    float max = 0;
    for (int i = 0; i < width * height; i++)
    {
        if (harris_response[i] > max)
            max = harris_response[i];
    }

    unsigned char *rgb_gray = new uint8_t[3 * width * height];
    for (int i = 0; i < width * height; i++)
    {
        float value = 255 * harris_response[i] / max;
        rgb_gray[i * 3] = value;
        rgb_gray[i * 3 + 1] = value;
        rgb_gray[i * 3 + 2] = value;
    }

    stbi_write_png("harris_response.png", width, height, 3, rgb_gray, 3 * width);
    
    delete[] harris_response;
    delete[] rgb_gray;

    return 0;
}