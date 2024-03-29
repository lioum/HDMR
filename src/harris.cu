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

__global__ void grayscale(unsigned char *src, float *dst, int height, int width, int spitch, int dpitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    

    float r = src[y * spitch + 3 * x];
    float g = src[y * spitch + 3 * x + 1];
    float b = src[y * spitch + 3 * x + 2];

    dst[y * dpitch + x] = 0.299 * r + 0.587 * g + 0.114 * b;
}

__global__ void multiply_ew(float *lhs, float *rhs, float *dst, int height,
                            int width, int spitch, int dpitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int si = y * spitch + x;

    dst[y * dpitch + x] = lhs[si] * rhs[si];
}

__device__ void convolve2D(float *src, float *kernel, float *dst, int height,
                           int width, int mask_size, float *s_src, int spitch,
                           int dpitch)
{
    int block_x = blockIdx.x * blockDim.x;
    int block_y = blockIdx.y * blockDim.y;

    int x = block_x + threadIdx.x;
    int y = block_y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    
    size_t tile_w = blockDim.x + mask_size - 1;
    size_t tile_h = blockDim.y + mask_size - 1;
    int half_size = mask_size / 2;
    for (int y = threadIdx.y; y < tile_h; y += blockDim.y)
    {
        for (int x = threadIdx.x; x < tile_w; x += blockDim.x)
        {
            int u = x + block_x - half_size;
            int v = y + block_y - half_size;
            if (u < 0 || u >= width || v < 0 || v >= height)
                s_src[y * tile_w + x] = 0; 
            else
                s_src[y * tile_w + x] = src[v * spitch + u];
        }
    }
    __syncthreads();

    float Pvalue = 0;
    for (int k = 0; k < mask_size; k++)
    {
        for (int j = 0; j < mask_size; j++)
        {
            int u = threadIdx.x + 2 * half_size - j;
            int v = threadIdx.y + 2 * half_size - k;
            Pvalue += s_src[v * tile_w + u] * kernel[k * mask_size + j];                
        }
    }

    dst[y * dpitch + x] = Pvalue;
}

__global__ void convolve_gauss(float *src, float *dst, int height, int width,
                               int spitch, int dpitch)
{
    extern __shared__ float s_src[];
    convolve2D(src, d_gauss, dst, height, width, ker_size, s_src, spitch, dpitch);
}

__global__ void convolve_gauss_dx(float *src, float *dst, int height, int width,
                                  int spitch, int dpitch)
{
    extern __shared__ float s_src[];
    convolve2D(src, d_gauss_dx, dst, height, width, ker_size, s_src, spitch, dpitch);
}

__global__ void convolve_gauss_dy(float *src, float *dst, int height, int width,
                                  int spitch, int dpitch)
{
    extern __shared__ float s_src[];
    convolve2D(src, d_gauss_dy, dst, height, width, ker_size, s_src, spitch, dpitch);
}

__global__ void compute_response(float *Wxx, float *Wyy, float *Wxy, float *dst,
                                 int height, int width, int spitch, int dpitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    
    int i = y * spitch + x;

    float Wdet = Wxx[i] * Wyy[i] - Wxy[i] * Wxy[i];
    float Wtr = Wxx[i] + Wyy[i];

    dst[y * dpitch + x] = Wdet / (Wtr + 1);
}

__global__ void thresh_harris(float *d_harris_response, float *d_min_coef,
                              float *d_max_coef, float threshold, size_t height,
                              size_t width, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    d_harris_response[y * pitch + x] = d_harris_response[y * pitch + x]
        * (d_harris_response[y * pitch + x]
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

__global__ void get_local_maximums(float *mat, float *dst, int height, int width, int pitch)
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
            if (max <= mat[i * pitch + j])
                max = mat[i * pitch + j];
        }
    }

    dst[y * width + x] = max * is_close(mat[y * pitch + x], max);
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
                          int height, size_t harris_pitch)
{
    unsigned char *d_img_rgb; 
    size_t img_rgb_pitch, img_gray_pitch;
    cudaMallocPitch(&d_img_rgb, &img_rgb_pitch, 3 * width * sizeof(unsigned char), height);
    float *d_img_gray;        
    cudaMallocPitch(&d_img_gray, &img_gray_pitch, width * sizeof(float), height);


    cudaMemcpy2D(
        d_img_rgb, img_rgb_pitch, img_rgb, 3 * width * sizeof(unsigned char),
        3 * width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);
    
    
    cudaMemcpyToSymbol(d_gauss, gauss, sizeof(gauss), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_gauss_dx, gauss_dx, sizeof(gauss_dx), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_gauss_dy, gauss_dy, sizeof(gauss_dy), 0, cudaMemcpyHostToDevice);

    dim3 threads(32,32);
    dim3 blocks((width+threads.x-1)/threads.x,
                (height+threads.y-1)/threads.y);
    size_t padding = ker_size - 1;
    size_t shared_size =
        (threads.x + padding) * (threads.y + padding) * sizeof(float);

    grayscale<<<blocks, threads>>>(d_img_rgb, d_img_gray, height, width, img_rgb_pitch, img_gray_pitch / sizeof(float));

    cudaDeviceSynchronize();

    float *d_imx, *d_imy, *d_Wxx, *d_Wyy, *d_Wxy;
    size_t imx_pitch, imy_pitch, Wxx_pitch, Wyy_pitch, Wxy_pitch;

    cudaMallocPitch(&d_imx, &imx_pitch, width * sizeof(float), height);
    cudaMallocPitch(&d_imy, &imy_pitch, width * sizeof(float), height);

    cudaMallocPitch(&d_Wxx, &Wxx_pitch, width * sizeof(float), height);
    cudaMallocPitch(&d_Wyy, &Wyy_pitch, width * sizeof(float), height);
    cudaMallocPitch(&d_Wxy, &Wxy_pitch, width * sizeof(float), height);

    cudaDeviceSynchronize();

    convolve_gauss_dx<<<blocks, threads, shared_size>>>(
        d_img_gray, d_imx, height, width, img_gray_pitch / sizeof(float),
        imx_pitch / sizeof(float));
    convolve_gauss_dy<<<blocks, threads, shared_size>>>(
        d_img_gray, d_imy, height, width, img_gray_pitch / sizeof(float),
        imy_pitch / sizeof(float));

    cudaDeviceSynchronize();

    float *d_imx2, *d_imy2, *d_imxy;
    size_t imx2_pitch, imy2_pitch, imxy_pitch;
    cudaMallocPitch(&d_imx2, &imx2_pitch, width * sizeof(float), height);
    cudaMallocPitch(&d_imy2, &imy2_pitch, width * sizeof(float), height);
    cudaMallocPitch(&d_imxy, &imxy_pitch, width * sizeof(float), height);

    multiply_ew<<<blocks, threads, shared_size>>>(
        d_imx, d_imx, d_imx2, height, width, imx_pitch / sizeof(float),
        imx2_pitch / sizeof(float));
    multiply_ew<<<blocks, threads, shared_size>>>(
        d_imy, d_imy, d_imy2, height, width, imx_pitch / sizeof(float),
        imy2_pitch / sizeof(float));
    multiply_ew<<<blocks, threads, shared_size>>>(
        d_imx, d_imy, d_imxy, height, width, imx_pitch / sizeof(float),
        imxy_pitch / sizeof(float));

    cudaDeviceSynchronize();

    cudaFree(d_imx);
    cudaFree(d_imy);

    convolve_gauss<<<blocks, threads, shared_size>>>(
        d_imx2, d_Wxx, height, width, imx2_pitch / sizeof(float),
        Wxx_pitch / sizeof(float));
    convolve_gauss<<<blocks, threads, shared_size>>>(
        d_imy2, d_Wyy, height, width, imy2_pitch / sizeof(float),
        Wyy_pitch / sizeof(float));
    convolve_gauss<<<blocks, threads, shared_size>>>(
        d_imxy, d_Wxy, height, width, imxy_pitch / sizeof(float),
        Wxy_pitch / sizeof(float));

    cudaDeviceSynchronize();

    compute_response<<<blocks, threads>>>(
        d_Wxx, d_Wyy, d_Wxy, d_harris_response, height, width,
        Wxx_pitch / sizeof(float), harris_pitch / sizeof(float));

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

    float *d_harris_response, *d_harris_unpitched;
    size_t harris_pitch;
    cudaMalloc(&d_harris_unpitched, size * sizeof(float));
    cudaMallocPitch(&d_harris_response, &harris_pitch, width * sizeof(float), height);

    compute_harris_response(img_rgb, d_harris_response, width, height, harris_pitch);
    
    cudaMemcpy2D(d_harris_unpitched, width * sizeof(float), d_harris_response,
                 harris_pitch, width * sizeof(float), height,
                 cudaMemcpyDeviceToDevice);

    float *d_min_coef, *d_max_coef;
    cudaMalloc(&d_min_coef, sizeof(float));
    cudaMalloc(&d_max_coef, sizeof(float));

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0; 
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                           d_harris_unpitched, d_min_coef, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                           d_harris_unpitched, d_min_coef, size);
    
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                           d_harris_unpitched, d_max_coef, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                           d_harris_unpitched, d_max_coef, size);

    cudaDeviceSynchronize();

    thresh_harris<<<blocks, threads>>>(d_harris_response, d_min_coef,
                                       d_max_coef, threshold, height, width,
                                       harris_pitch / sizeof(float));

    cudaDeviceSynchronize();
 
    int *d_indices, *d_sorted_indices;
    cudaMalloc(&d_indices, size * sizeof(int));
    cudaMalloc(&d_sorted_indices, size * sizeof(int));
    init_indices<<<(nb_threads + size - 1) / nb_threads, nb_threads>>>(d_indices, size);

    float *d_local_maxs, *d_sorted_maxs;
    cudaMalloc(&d_local_maxs, size * sizeof(float));
    cudaMalloc(&d_sorted_maxs, size * sizeof(float));

    get_local_maximums<<<blocks, threads>>>(d_harris_response, d_local_maxs,
                                            height, width, harris_pitch / sizeof(float));

    cudaDeviceSynchronize();
    cudaFree(d_harris_response);
    cudaFree(d_min_coef);
    cudaFree(d_max_coef);
    cudaFree(d_harris_unpitched);

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

__global__ void mark_points(unsigned char *img_rgb, int *x_coords,
                            int *y_coords, int height, int width,
                            int coords_size, int mask_size, int img_pitch)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.z * blockIdx.z + threadIdx.z;
    
    if (i >= coords_size)
        return;
    
    int y = y_coords[i];
    int x = x_coords[i];

    int half_size = mask_size / 2;
    int u = x - (j - half_size), v = y - (k - half_size);

    if (u < 0 || u >= width || v < 0 || v >= height)
        return;
    
    img_rgb[v * img_pitch + 3 * u] = 0;
    img_rgb[v * img_pitch + 3 * u + 1] = 0;
    img_rgb[v * img_pitch + 3 * u + 2] = 255;
}

void color_image(unsigned char *img_rgb, int *x_coords, int *y_coords, int height, int width, int coords_size) {
    
    unsigned char *d_img_rgb;
    size_t img_pitch;
    int *d_x_coords, *d_y_coords;
    
    cudaMallocPitch(&d_img_rgb, &img_pitch, 3 * width * sizeof(unsigned char), height);
    cudaMalloc(&d_x_coords, coords_size * sizeof(int));
    cudaMalloc(&d_y_coords, coords_size * sizeof(int));

    cudaMemcpy2D(
        d_img_rgb, img_pitch, img_rgb, 3 * width * sizeof(unsigned char),
        3 * width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_coords, x_coords, coords_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_coords, y_coords, coords_size * sizeof(int), cudaMemcpyHostToDevice);
    
    int mask_size = 3;
    
    dim3 threads(1024 / (mask_size * mask_size), mask_size, mask_size);
    dim3 blocks((coords_size + threads.x - 1) / threads.x, 1, 1);

    mark_points<<<blocks, threads>>>(d_img_rgb, d_x_coords, d_y_coords, height,
                                     width, coords_size, mask_size, img_pitch);

    cudaMemcpy2D(img_rgb, 3 * width * sizeof(unsigned char), d_img_rgb,
                 img_pitch, 3 * width * sizeof(unsigned char), height,
                 cudaMemcpyDeviceToHost);
    
    cudaFree(d_img_rgb);
    cudaFree(d_x_coords);
    cudaFree(d_y_coords);
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

    detect_harris_points(img, x_coords, y_coords, cornerness, 0.01, width, height, nb_keypoints);

    if (argc == 2) {
        for (size_t i = 0; i < nb_keypoints; i++)
        {
            if (cornerness[i] == 0)
                break;
            std::cout << y_coords[i] << "," << x_coords[i] << "," << cornerness[i] << std::endl;
        }
    }
    else {
        auto nb_points = std::find(cornerness, cornerness + nb_keypoints, 0.0) - cornerness;
        color_image(img, x_coords, y_coords, height, width, nb_points);
        stbi_write_jpg(argv[2], width, height, 3, img, 60);
    }
    
    stbi_image_free(img);

    delete[] x_coords;
    delete[] y_coords;
    delete[] cornerness;

    return 0;
}
