#pragma once

//transpose the result at the end
__launch_bounds__(256)
__global__ void Classical2_horizontalBlur_Kernel(float* dst, float* src, int64_t w, int64_t h, float* gaussiankernel, int gaussiansize){
    int64_t x = threadIdx.x + blockIdx.x*blockDim.x;
    int64_t size = w*h;
    if (x >= size) return;

    int current_line = x/w;

    float weight = 0.0f;

    float out = 0.0f;
    for (int i = max(x-gaussiansize, current_line*w); i <= min(x+gaussiansize, (current_line+1)*w-1); i++){
        const float gauss = gaussiankernel[gaussiansize+i-x];
        out += src[i]*gauss;
        weight += gauss;
    }
    dst[x] = out/weight;
}

//best is to use 8x32 rectangle 
__launch_bounds__(256)
__global__ void Classical2_verticalBlur_Kernel(float* dst, float* src, int64_t w, int64_t h, float* gaussiankernel, int gaussiansize){
    int64_t x = threadIdx.x + blockIdx.x*blockDim.x;
    int64_t y = threadIdx.y + blockIdx.y*blockDim.y;
    if (x >= w) return;
    if (y >= h) return;

    float weight = 0.0f;
    float out = 0.0f;
    for (int64_t i = max(y-gaussiansize, (int64_t)0); i <= min(y+gaussiansize, h-1); i++){
        //if (x == 423 && y == 323) printf("%f at %d for %f\n", gaussiankernel[gaussiansize+i-y], gaussiansize+i-y, src[i*w+x]);
        out += src[i * w + x]*gaussiankernel[gaussiansize+i-y];
        weight += gaussiankernel[gaussiansize+i-y];
    }
    dst[y*w+x] = out/weight;
}

struct Classical2Gaussian{
    TestingPlane& plane;
    float sigma;
    float* temp;
    float* gaussiankernel_d;
    int windowsize;
    Classical2Gaussian(TestingPlane& plane, float sigma, int window_multiplier = 3) : plane(plane), sigma(sigma) {
        windowsize = sigma*window_multiplier;
        hipMalloc(&temp, sizeof(float)*plane.width*plane.height);
        hipMalloc(&gaussiankernel_d, sizeof(float)*(2*windowsize+1));
        loadGaussianKernel<<<dim3(1), dim3(2*windowsize+1)>>>(gaussiankernel_d, windowsize, sigma);
        hipMemcpyDtoD(temp, plane.plane_d, sizeof(float)*plane.width*plane.height);
        hipDeviceSynchronize();
    }
    ~Classical2Gaussian(){
        hipFree(temp);
        hipFree(gaussiankernel_d);
    }
    void sync(){
        hipDeviceSynchronize();
    }
    void run(){
        int64_t wh = plane.width*plane.height;
        int64_t th_x = std::min((int64_t)256, wh);
        int64_t bl_x = (wh-1)/th_x + 1;

        int64_t verticalth_x = 8;
        int64_t verticalth_y = 32;
        int64_t verticalbl_x = (plane.width-1)/verticalth_x+1;
        int64_t verticalbl_y = (plane.height-1)/verticalth_y+1;
        Classical2_horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x)>>>(temp, plane.plane_d, plane.width, plane.height, gaussiankernel_d, windowsize);
        Classical2_verticalBlur_Kernel<<<dim3(verticalbl_x, verticalbl_y), dim3(verticalth_x, verticalth_y)>>>(plane.plane_d, temp, plane.width, plane.height, gaussiankernel_d, windowsize);
    }
};