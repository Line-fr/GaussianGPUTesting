#pragma once

__global__ void loadGaussianKernel(float* gaussiankernel, int gaussiansize, double sigma){
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x > 2*gaussiansize) return;
    gaussiankernel[x] = expf(-(gaussiansize-x)*(gaussiansize-x)/(2*sigma*sigma))/(sqrt(TAU*sigma*sigma));
}

//transpose the result at the end
__launch_bounds__(256)
__global__ void Classical_horizontalBlur_Kernel(float* dst, float* src, int w, int h, float* gaussiankernel, int gaussiansize){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int size = w*h;
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

__launch_bounds__(256)
__global__ void Classical_verticalBlur_Kernel(float* dst, float* src, int w, int h, float* gaussiankernel, int gaussiansize){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int size = w*h;
    if (x >= size) return;

    int current_line = x/w;
    int current_column = x%w;

    float weight = 0.0f;

    float out = 0.0f;
    for (int i = max(current_line-gaussiansize, 0); i <= min(current_line+gaussiansize, h-1); i++){
        out += src[i * w + current_column]*gaussiankernel[gaussiansize+i-current_line];
        weight += gaussiankernel[gaussiansize+i-current_line];
        //if (threadIdx.x == 0) printf("%f at %d\n", gaussiankernel[gaussiansize+i-current_line], gaussiansize+i-current_line);
    }
    dst[x] = out/weight;
}

struct ClassicalGaussian{
    TestingPlane& plane;
    float sigma;
    float* temp;
    float* gaussiankernel_d;
    int windowsize;
    ClassicalGaussian(TestingPlane& plane, float sigma, int window_multiplier = 3) : plane(plane), sigma(sigma) {
        windowsize = sigma*window_multiplier;
        hipMalloc(&temp, sizeof(float)*plane.width*plane.height);
        hipMalloc(&gaussiankernel_d, sizeof(float)*(2*windowsize+1));
        loadGaussianKernel<<<dim3(1), dim3(2*windowsize+1)>>>(gaussiankernel_d, windowsize, sigma);
        hipMemcpyDtoD(temp, plane.plane_d, sizeof(float)*plane.width*plane.height);
        hipDeviceSynchronize();
    }
    ~ClassicalGaussian(){
        hipFree(temp);
        hipFree(gaussiankernel_d);
    }
    void sync(){
        hipDeviceSynchronize();
    }
    void run(){
        int64_t wh = plane.width*plane.height;
        int th_x = std::min((int64_t)256, wh);
        int bl_x = (wh-1)/th_x + 1;
        Classical_horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x)>>>(temp, plane.plane_d, plane.width, plane.height, gaussiankernel_d, windowsize);
        Classical_verticalBlur_Kernel<<<dim3(bl_x), dim3(th_x)>>>(plane.plane_d, temp, plane.width, plane.height, gaussiankernel_d, windowsize);
    }
};