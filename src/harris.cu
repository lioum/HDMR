#include <cub/cub.cuh> 

#include "stb_image.h"
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

__global__ void thresh_harris(float *d_harris_response, float *d_min_coef,
                              float *d_max_coef, float threshold, size_t size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < 0 || x >= size)
        return;

    d_harris_response[x] = d_harris_response[x] * (d_harris_response[x]
        > (*d_min_coef + threshold * (*d_max_coef - *d_min_coef)));
}

__global__ void init_indices(int *d_indices, size_t size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < 0 || x >= size)
        return;

    d_indices[x] = x;
}

__device__ bool is_close(double a, double b, double rtol = 1e-5, double atol = 1e-8)
{
    return abs(a - b) <= (atol + rtol * abs(b));
}

__global__ void get_local_maximums(float *mat, float *dst, int height, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    
    int block_i = y - (y > 0);
    int block_j = x - (x > 0);
    int block_h = 1 + (y > 0) + (y < height - 1);
    int block_w = 1 + (x > 0) + (x < width - 1);

    float max = 0;
    for (int i = block_i; i < block_i + block_h; i++)
    {
        for (int j = block_j; j < block_j + block_w; j++)
        {
            if (max <= mat[i * width + j])
                max = mat[i * width + j];
        }
    }
    dst[y * width + x] = max * is_close(mat[y * width + x], max);
}

__global__ void compute_coords(int *sorted_indices, int *d_x_coords, int *d_y_coords, int width, size_t nb_keypoints)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= nb_keypoints)
        return;

    d_y_coords[x] = sorted_indices[x] / width;
    d_x_coords[x] = sorted_indices[x] % width;
}

void compute_harris_response(const unsigned char *img_rgb, float *d_harris_response, int width,
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
        (x_derivative_ker_size + threads.x - 1) / threads.x,
        (x_derivative_ker_size + threads.y - 1) / threads.y);

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
        d_Wxx, d_Wyy, d_Wxy, d_harris_response, size);

    cudaDeviceSynchronize();

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
}

void detect_harris_points(const unsigned char *img_rgb, int *x_coords,
                          int *y_coords, float *cornerness, float threshold,
                          int width, int height, size_t derivative_ker_size,
                          size_t opening_size, size_t nb_keypoints)
{
    size_t size = width * height;
    dim3 threads(32,32);
    int nb_threads = threads.x * threads.y;
    dim3 blocks((width+threads.x-1)/threads.x,
                (height+threads.y-1)/threads.y);   

    float *d_harris_response;
    cudaMalloc(&d_harris_response, size * sizeof(float));

    compute_harris_response(img_rgb, d_harris_response, width, height,
                            derivative_ker_size, opening_size);

    float *d_min_coef, *d_max_coef;
    cudaMalloc(&d_min_coef, sizeof(float));
    cudaMalloc(&d_max_coef, sizeof(float));

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0; 
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                           d_harris_response, d_min_coef, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                           d_harris_response, d_min_coef, size);
    
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                           d_harris_response, d_max_coef, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                           d_harris_response, d_max_coef, size);

    cudaDeviceSynchronize();

    thresh_harris<<<(nb_threads + size - 1) / nb_threads, nb_threads>>>(
        d_harris_response, d_min_coef, d_max_coef, threshold, size);
    
    cudaDeviceSynchronize();
 
    int *d_indices, *d_sorted_indices;
    cudaMalloc(&d_indices, size * sizeof(int));
    cudaMalloc(&d_sorted_indices, size * sizeof(int));
    init_indices<<<(nb_threads + size - 1) / nb_threads, nb_threads>>>(d_indices, size);

    float *d_local_maxs, *d_sorted_maxs;
    cudaMalloc(&d_local_maxs, size * sizeof(float));
    cudaMalloc(&d_sorted_maxs, size * sizeof(float));

    get_local_maximums<<<blocks, threads>>>(d_harris_response, d_local_maxs, height, width);

    cudaDeviceSynchronize();
    cudaFree(d_harris_response);
    cudaFree(d_min_coef);
    cudaFree(d_max_coef);

    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, d_local_maxs, d_sorted_maxs,
        d_indices, d_sorted_indices, size);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, d_local_maxs, d_sorted_maxs,
        d_indices, d_sorted_indices, size);
    
    cudaDeviceSynchronize();

    int *d_x_coords, *d_y_coords;
    cudaMalloc(&d_x_coords, nb_keypoints * sizeof(int));
    cudaMalloc(&d_y_coords, nb_keypoints * sizeof(int));

    compute_coords<<<(nb_threads + nb_keypoints - 1) / nb_threads,
                     nb_threads>>>(d_sorted_indices, d_x_coords, d_y_coords,
                                   width, nb_keypoints);

    cudaDeviceSynchronize();

    cudaMemcpy(x_coords, d_x_coords, nb_keypoints * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(y_coords, d_y_coords, nb_keypoints * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(cornerness, d_sorted_maxs, nb_keypoints * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_indices);
    cudaFree(d_sorted_indices);

    cudaFree(d_local_maxs);
    cudaFree(d_sorted_maxs);
    cudaFree(d_temp_storage);
}

int main(int argc, char *argv[])
{
    if (argc <= 1)
        return 2;

    int width, height, bpp;
    auto img = stbi_load(argv[1], &width, &height, &bpp, 3);
    
    size_t nb_keypoints = 2000;
    int *x_coords = new int[nb_keypoints];
    int *y_coords = new int[nb_keypoints];
    float *cornerness = new float[nb_keypoints];

    detect_harris_points(img, x_coords, y_coords, cornerness, 0.1, width, height,
                         3, 3, nb_keypoints);

    stbi_image_free(img);

    for (size_t i = 0; i < nb_keypoints; i++)
    {
        if (cornerness[i] == 0)
            break;
        std::cout << y_coords[i] << ", " << x_coords[i] << std::endl;
    }
    
    delete[] x_coords;
    delete[] y_coords;
    delete[] cornerness;

    return 0;
}
