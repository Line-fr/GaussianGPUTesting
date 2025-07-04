#pragma once

__launch_bounds__(256)
__global__ void Tiled3GaussianBlur_Kernel(float* src, float* dst, int64_t width, int64_t height, float* gaussiankernel){
    int64_t originalBl_X = blockIdx.x;
    //let's determine which scale our block is in and adjust our input parameters accordingly

    const int64_t blockwidth = (width-1)/48+1;

    const int64_t x = threadIdx.x + 48*(originalBl_X%blockwidth);
    const int64_t y = threadIdx.y + 48*(originalBl_X/blockwidth);
    const int64_t thx = threadIdx.x;
    const int64_t thy = threadIdx.y;

    __shared__ float tampon[64*64]; //we import into tampon, compute onto tampon and then put into dst
    //tampon has 8 of border on each side with no thread
    const int tampon_base_x = x - thx - 8;
    const int tampon_base_y = y - thy - 8;

    //fill tampon
    #pragma unroll
    for (int region_x = 0; region_x < 64; region_x += 16){
        #pragma unroll
        for (int region_y = 0; region_y < 64; region_y += 16){
            tampon[(thy+region_y)*64+thx+region_x] = (tampon_base_x + thx +region_x >= 0 && tampon_base_x + thx +region_x < width && tampon_base_y + thy + region_y >= 0 && tampon_base_y + thy + region_y < height) ? src[(tampon_base_y+thy+region_y)*width + tampon_base_x+thx+region_x] : 0.f;
        }
    }
    __syncthreads();

    //horizontalBlur on tampon restraint into rectangle [8 - 56][0 - 64] -> 12 pass per thread
    float tot[3] = {0.f, 0.f, 0.f};
    float out[3][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

    #pragma unroll
    for (int region_x = 0; region_x < 3; region_x++){

        //border handling precompute
        for (int i = 0; i < 17; i++){
            if (tampon_base_x+thx+i+region_x*16 >= 0 && tampon_base_x+thx+i+region_x*16 < width) tot[region_x] += gaussiankernel[i];
        }

        #pragma unroll
        for (int region_y = 0; region_y < 4; region_y++){
            for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
                out[region_x][region_y] += tampon[(thy+region_y*16)*64 + thx+i+region_x*16]*gaussiankernel[i];
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int region_x = 0; region_x < 3; region_x++){
        #pragma unroll
        for (int region_y = 0; region_y < 4; region_y++){
            tampon[(thy+region_y*16)*64 + thx+8+region_x*16] = out[region_x][region_y]/tot[region_x];
        }
    }
    __syncthreads();

    //verticalBlur on tampon restraint into rectangle [8 - 56][8 - 56] -> 9 pass per thread
    #pragma unroll
    for (int region_y = 0; region_y < 3; region_y++){
        
        tot[region_y] = 0;
        //border handling precompute
        for (int i = 0; i < 17; i++){
            if (tampon_base_y+thy+i+region_y*16 >= 0 && tampon_base_y+thy+i+region_y*16 < height) tot[region_y] += gaussiankernel[i];
        }

        #pragma unroll
        for (int region_x = 0; region_x < 3; region_x++){
            out[region_y][region_x] = 0;
            for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
                out[region_y][region_x] += tampon[(thy+i+region_y*16)*64 + thx+8+region_x*16]*gaussiankernel[i];
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int region_y = 0; region_y < 3; region_y++){
        #pragma unroll
        for (int region_x = 0; region_x < 3; region_x++){
            tampon[(thy+8+region_y*16)*64 + thx+8+region_x*16] = out[region_y][region_x]/tot[region_y];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int region_x = 0; region_x < 3; region_x++){
        #pragma unroll
        for (int region_y = 0; region_y < 3; region_y++){
            if (tampon_base_x + thx +8+region_x*16 >= 0 && tampon_base_x + thx +8+region_x*16 < width && tampon_base_y + thy +8+region_y*16 >= 0 && tampon_base_y + thy +8+region_y*16 < height) dst[(tampon_base_y+thy+8+region_y*16)*width + tampon_base_x+thx+8+region_x*16] = tampon[(thy+8+region_y*16)*64+thx+8+region_x*16];
        }
    }
}

struct Tiled3Gaussian{
    TestingPlane& plane;
    float sigma;
    float* temp; //we will not work on live plane, we will copy at sync
    float* temp2;
    float* gaussiankernel_d;
    int oscillate = 0; //indicate the plane the correct data sits on
    Tiled3Gaussian(TestingPlane& plane, float sigma) : plane(plane), sigma(sigma) {
        const int windowsize = 8;
        hipMalloc(&temp, sizeof(float)*plane.width*plane.height);
        hipMalloc(&temp2, sizeof(float)*plane.width*plane.height);
        hipMalloc(&gaussiankernel_d, sizeof(float)*(2*windowsize+1));
        loadGaussianKernel<<<dim3(1), dim3(2*windowsize+1)>>>(gaussiankernel_d, windowsize, sigma);
        float* temps[2] = {temp, temp2};
        hipMemcpyDtoD(temps[oscillate], plane.plane_d, sizeof(float)*plane.width*plane.height);
        hipDeviceSynchronize();
    }
    ~Tiled3Gaussian(){
        hipFree(temp);
        hipFree(temp2);
        hipFree(gaussiankernel_d);
    }
    void sync(){
        float* temps[2] = {temp, temp2};
        hipMemcpyDtoD(plane.plane_d, temps[oscillate], sizeof(float)*plane.width*plane.height);
        hipDeviceSynchronize();
    }
    void run(){
        float* temps[2] = {temp, temp2};
        float* source = temps[oscillate];
        float* dest = temps[oscillate^1];

        int64_t th_x = 16;
        int64_t th_y = 16;
        int64_t bl_x = (plane.width-1)/(3*th_x)+1;
        int64_t bl_y = (plane.height-1)/(3*th_y)+1;

        Tiled3GaussianBlur_Kernel<<<dim3(bl_x*bl_y), dim3(th_x, th_y)>>>(source, dest, plane.width, plane.height, gaussiankernel_d);

        oscillate ^= 1; //we moved the result in the other place
    }
};