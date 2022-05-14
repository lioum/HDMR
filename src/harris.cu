#include <cub/cub.cuh> 

#include "stb_image.h"
#include "stb_image_write.h"

float gauss[] = {
    1.02798843e-04, 1.31755659e-03, 6.08748501e-03, 1.01389764e-02,
    6.08748501e-03, 1.31755659e-03, 1.02798843e-04, 1.31755659e-03,
    1.68869153e-02, 7.80223366e-02, 1.29949664e-01, 7.80223366e-02,
    1.68869153e-02, 1.31755659e-03, 6.08748501e-03, 7.80223366e-02,
    3.60485318e-01, 6.00404295e-01, 3.60485318e-01, 7.80223366e-02,
    6.08748501e-03, 1.01389764e-02, 1.29949664e-01, 6.00404295e-01,
    1.00000000e+00, 6.00404295e-01, 1.29949664e-01, 1.01389764e-02,
    6.08748501e-03, 7.80223366e-02, 3.60485318e-01, 6.00404295e-01,
    3.60485318e-01, 7.80223366e-02, 6.08748501e-03, 1.31755659e-03,
    1.68869153e-02, 7.80223366e-02, 1.29949664e-01, 7.80223366e-02,
    1.68869153e-02, 1.31755659e-03, 1.02798843e-04, 1.31755659e-03,
    6.08748501e-03, 1.01389764e-02, 6.08748501e-03, 1.31755659e-03,
    1.02798843e-04
};

float gauss_dx[] = {
    3.08396530e-04,  2.63511317e-03,  6.08748501e-03,  0.00000000e+00,
    -6.08748501e-03, -2.63511317e-03, -3.08396530e-04, 3.95266976e-03,
    3.37738305e-02,  7.80223366e-02,  0.00000000e+00,  -7.80223366e-02,
    -3.37738305e-02, -3.95266976e-03, 1.82624550e-02,  1.56044673e-01,
    3.60485318e-01,  0.00000000e+00,  -3.60485318e-01, -1.56044673e-01,
    -1.82624550e-02, 3.04169293e-02,  2.59899329e-01,  6.00404295e-01,
    0.00000000e+00,  -6.00404295e-01, -2.59899329e-01, -3.04169293e-02,
    1.82624550e-02,  1.56044673e-01,  3.60485318e-01,  0.00000000e+00,
    -3.60485318e-01, -1.56044673e-01, -1.82624550e-02, 3.95266976e-03,
    3.37738305e-02,  7.80223366e-02,  0.00000000e+00,  -7.80223366e-02,
    -3.37738305e-02, -3.95266976e-03, 3.08396530e-04,  2.63511317e-03,
    6.08748501e-03,  0.00000000e+00,  -6.08748501e-03, -2.63511317e-03,
    -3.08396530e-04
};

float gauss_dy[] = {
    3.08396530e-04,  3.95266976e-03,  1.82624550e-02,  3.04169293e-02,
    1.82624550e-02,  3.95266976e-03,  3.08396530e-04,  2.63511317e-03,
    3.37738305e-02,  1.56044673e-01,  2.59899329e-01,  1.56044673e-01,
    3.37738305e-02,  2.63511317e-03,  6.08748501e-03,  7.80223366e-02,
    3.60485318e-01,  6.00404295e-01,  3.60485318e-01,  7.80223366e-02,
    6.08748501e-03,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    -6.08748501e-03, -7.80223366e-02, -3.60485318e-01, -6.00404295e-01,
    -3.60485318e-01, -7.80223366e-02, -6.08748501e-03, -2.63511317e-03,
    -3.37738305e-02, -1.56044673e-01, -2.59899329e-01, -1.56044673e-01,
    -3.37738305e-02, -2.63511317e-03, -3.08396530e-04, -3.95266976e-03,
    -1.82624550e-02, -3.04169293e-02, -1.82624550e-02, -3.95266976e-03,
    -3.08396530e-04
};

const size_t ker_size = 7;

__constant__ float d_gauss[ker_size * ker_size];
__constant__ float d_gauss_dx[ker_size * ker_size];
__constant__ float d_gauss_dy[ker_size * ker_size];

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

__device__ void convolve2D(float *src, float *kernel, float *dst, int height, int width, int mask_size, float *s_src)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    
    s_src[threadIdx.y * blockDim.x + threadIdx.x] = src[y * width + x];
    __syncthreads();

    int x_min = blockIdx.x * blockDim.x;
    int x_max = (blockIdx.x + 1) * blockDim.x;
    int y_min = blockIdx.y * blockDim.y;
    int y_max = (blockIdx.y + 1) * blockDim.y;

    float Pvalue = 0;
    int half_size = mask_size / 2;
    for (int k = 0; k < mask_size; k++)
    {
        for (int j = 0; j < mask_size; j++)
        {
            int u = x - (j - half_size), v = y - (k - half_size);
            
            if (u < 0 || u >= width || v < 0 || v >= height)
                continue;

            if (u < x_min || u >= x_max || v < y_min || v >= y_max)
                Pvalue += src[v * width + u] * kernel[k * mask_size + j];
            else
            {
                u = threadIdx.x - (j - half_size), v = threadIdx.y - (k - half_size);
                Pvalue += s_src[v * blockDim.x + u] * kernel[k * mask_size + j];                
            }
        }
    }

    dst[y * width + x] = Pvalue;
}

__global__ void convolve_gauss(float *src, float *dst, int height, int width) {
    extern __shared__ float s_src[];
    convolve2D(src, d_gauss, dst, height, width, ker_size, s_src);
}

__global__ void convolve_gauss_dx(float *src, float *dst, int height, int width) {
    extern __shared__ float s_src[];
    convolve2D(src, d_gauss_dx, dst, height, width, ker_size, s_src);
}

__global__ void convolve_gauss_dy(float *src, float *dst, int height, int width) {
    extern __shared__ float s_src[];
    convolve2D(src, d_gauss_dy, dst, height, width, ker_size, s_src);
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
                          int height)
{

    size_t size = width * height;
    
    unsigned char *d_img_rgb; 
    cudaMalloc(&d_img_rgb, 3 * size * sizeof(unsigned char));
    float *d_img_gray;        
    cudaMalloc(&d_img_gray, size * sizeof(float));

    cudaMemcpy(d_img_rgb, img_rgb, 3 * size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_gauss, gauss, ker_size * ker_size * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_gauss_dx, gauss_dx, ker_size * ker_size * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_gauss_dy, gauss_dy, ker_size * ker_size * sizeof(float), 0, cudaMemcpyHostToDevice);

    dim3 threads(32,32);
    dim3 blocks((width+threads.x-1)/threads.x,
                (height+threads.y-1)/threads.y);
    size_t shared_size = threads.x * threads.y * sizeof(float);

    grayscale<<<blocks, threads>>>(d_img_rgb, d_img_gray, height, width);
    cudaDeviceSynchronize();

    float *d_imx, *d_imy, *d_Wxx, *d_Wyy, *d_Wxy, *d_response;

    cudaMalloc(&d_imx, size * sizeof(float));
    cudaMalloc(&d_imy, size * sizeof(float));

    cudaMalloc(&d_Wxx, size * sizeof(float));
    cudaMalloc(&d_Wyy, size * sizeof(float));
    cudaMalloc(&d_Wxy, size * sizeof(float));

    cudaMalloc(&d_response, size * sizeof(float));

    cudaDeviceSynchronize();

    convolve_gauss_dx<<<blocks,threads, shared_size>>>(d_img_gray, d_imx, height, width);
    convolve_gauss_dy<<<blocks,threads, shared_size>>>(d_img_gray, d_imy, height, width);

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

    convolve_gauss<<<blocks,threads, shared_size>>>(d_imx2, d_Wxx, height, width);
    convolve_gauss<<<blocks,threads, shared_size>>>(d_imy2, d_Wyy, height, width);
    convolve_gauss<<<blocks,threads, shared_size>>>(d_imxy, d_Wxy, height, width);

    cudaDeviceSynchronize();

    compute_response<<<(nb_threads + size - 1) / nb_threads, nb_threads>>>(
        d_Wxx, d_Wyy, d_Wxy, d_harris_response, size);

    cudaDeviceSynchronize();

    cudaFree(d_img_gray);
    cudaFree(d_img_rgb);

    cudaFree(d_imx2);
    cudaFree(d_imy2);
    cudaFree(d_imxy);
    
    cudaFree(d_Wxx);
    cudaFree(d_Wyy);
    cudaFree(d_Wxy);
}

void detect_harris_points(const unsigned char *img_rgb, int *x_coords,
                          int *y_coords, float *cornerness, float threshold,
                          int width, int height, size_t nb_keypoints)
{
    size_t size = width * height;
    dim3 threads(32,32);
    int nb_threads = threads.x * threads.y;
    dim3 blocks((width+threads.x-1)/threads.x,
                (height+threads.y-1)/threads.y);   
    
    

    float *d_harris_response;
    cudaMalloc(&d_harris_response, size * sizeof(float));

    compute_harris_response(img_rgb, d_harris_response, width, height);

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

    detect_harris_points(img, x_coords, y_coords, cornerness, 0.1, width, height, nb_keypoints);

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
