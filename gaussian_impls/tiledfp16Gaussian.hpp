#pragma once

__global__ void loadfp16GaussianKernel(half2* gaussiankernel, half2* gaussiankernel_integral, int gaussiansize, double sigma){
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x > 2*gaussiansize) return;
    gaussiankernel[x].x = expf(-(gaussiansize-x)*(gaussiansize-x)/(2*sigma*sigma))/(sqrt(TAU*sigma*sigma));
    gaussiankernel[x].y = gaussiankernel[x].x;
    if (x == 0){
        float acc = 0.;
        gaussiankernel_integral[0].x = 0.f;
        gaussiankernel_integral[0].y = 0.f;
        for (int i = 0; i <= 2*gaussiansize+1; i++){
            gaussiankernel_integral[i].x = acc;
            gaussiankernel_integral[i].y = acc;
            acc += expf(-(gaussiansize-i)*(gaussiansize-i)/(2*sigma*sigma))/(sqrt(TAU*sigma*sigma));
        }
    }
}

__launch_bounds__(256)
__global__ void Tiledfp16GaussianBlur_Kernel(float* src, float* dst, int64_t width, int64_t height, half2* gaussiankernel, half2* gaussiankernel_integral){
    int64_t originalBl_X = blockIdx.x;
    //let's determine which scale our block is in and adjust our input parameters accordingly

    const int64_t blockwidth = (width-1)/32+1;

    const int64_t x = threadIdx.x*2 + 32*(originalBl_X%blockwidth);
    const int64_t y = threadIdx.y + 32*(originalBl_X/blockwidth);
    const int64_t thx = threadIdx.x;
    const int64_t thy = threadIdx.y;

    __shared__ half2 tampon[48*48]; //we import into tampon, compute onto tampon and then put into dst
    half2* tampon2 = tampon+24*48; //will store the other alignement
    //tampon has 8 of border on each side with no thread
    //const int tampon_base_x = x - thx*2 - 8;
    const int tampon_base_y = y - thy - 8;

    //fill tampon
    #pragma unroll
    for (int region_x = 0; region_x < 48; region_x += 16){
        #pragma unroll
        for (int region_y = 0; region_y < 48; region_y += 16){
            half2 newel;
            newel.x = (x-8 +region_x >= 0 && x-8 +region_x < width && y-8 + region_y >= 0 && y-8 + region_y < height) ? src[(y-8+region_y)*width + x-8+region_x] : 0.f;
            newel.y = (x-7 +region_x >= 0 && x-7 +region_x < width && y-8 + region_y >= 0 && y-8 + region_y < height) ? src[(y-8+region_y)*width + x-7+region_x] : 0.f;
            tampon[(thy+region_y)*24+thx+region_x/2] = newel;
            newel.x = newel.y;
            newel.y = (x-6 +region_x >= 0 && x-6 +region_x < width && y-8 + region_y >= 0 && y-8 + region_y < height) ? src[(y-8+region_y)*width + x-6+region_x] : 0.f;
            tampon2[(thy+region_y)*24+thx+region_x/2] = newel;
        }
    }
    __syncthreads();

    //horizontalBlur on tampon restraint into rectangle [8 - 40][0 - 48] -> 6 pass per thread
    half2 tot[2] = {{0.f,0.f}, {0.f,0.f}};
    half2 out[2][3] = {{{0.f,0.f}, {0.f,0.f}, {0.f,0.f}}, {{0.f,0.f}, {0.f,0.f}, {0.f,0.f}}};

    #pragma unroll
    for (int region_x = 0; region_x < 2; region_x++){

        //border handling precompute
        const int beg = max((long)0, x-8+region_x*16)-(x-8+region_x*16);
        const int end2 = min(width, x+10+region_x*16)-(x-7+region_x*16);
        
        if (beg != 0) tot[region_x].y = gaussiankernel[beg-1].y;
        if (end2+(x-7+region_x*16) == width) tot[region_x].x = gaussiankernel[end2].x;

        tot[region_x] += gaussiankernel_integral[end2] - gaussiankernel_integral[beg];

        #pragma unroll
        for (int region_y = 0; region_y < 3; region_y++){
            for (int i = 0; i < 9; i++){ //starting 8 to the left and going 8 to the right
                out[region_x][region_y] += tampon[(thy+region_y*16)*24 + thx+i+region_x*8]*gaussiankernel[2*i];
                if (i != 8) out[region_x][region_y] += tampon2[(thy+region_y*16)*24 + thx+i+region_x*8]*gaussiankernel[2*i+1];
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int region_x = 0; region_x < 2; region_x++){
        #pragma unroll
        for (int region_y = 0; region_y < 3; region_y++){
            tampon[(thy+region_y*16)*24 + thx+4+region_x*8] = out[region_x][region_y]/tot[region_x];
        }
    }
    __syncthreads();

    //verticalBlur on tampon restraint into rectangle [8 - 40][8 - 40] -> 4 pass per thread
    #pragma unroll
    for (int region_y = 0; region_y < 2; region_y++){

        const int beg = max((long)0, y-8+region_y*16)-(y-8+region_y*16);
        const int end2 = min(height, y+9+region_y*16)-(y-8+region_y*16);
        tot[region_y] = gaussiankernel_integral[end2] - gaussiankernel_integral[beg];

        #pragma unroll
        for (int region_x = 0; region_x < 2; region_x++){
            out[region_y][region_x].x = 0;
            out[region_y][region_x].y = 0;
            for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
                out[region_y][region_x] += tampon[(thy+i+region_y*16)*24 + thx+4+region_x*8]*gaussiankernel[i];
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int region_y = 0; region_y < 2; region_y++){
        #pragma unroll
        for (int region_x = 0; region_x < 2; region_x++){
            tampon[(thy+8+region_y*16)*24 + thx+4+region_x*8] = out[region_y][region_x]/tot[region_y];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int region_x = 0; region_x < 2; region_x++){
        #pragma unroll
        for (int region_y = 0; region_y < 2; region_y++){
            if (x+region_x*16 >= 0 && x+region_x*16 < width && y+region_y*16 >= 0 && y+region_y*16 < height) dst[(y+region_y*16)*width + x+region_x*16] = tampon[(thy+8+region_y*16)*24+thx+4+region_x*8].x;
            if (x+1+region_x*16 >= 0 && x+1+region_x*16 < width && y+region_y*16 >= 0 && y+region_y*16 < height) dst[(y+region_y*16)*width + x+1+region_x*16] = tampon[(thy+8+region_y*16)*24+thx+4+region_x*8].y;
        }
    }
}

struct Tiledfp16Gaussian{
    TestingPlane& plane;
    float sigma;
    float* temp; //we will not work on live plane, we will copy at sync
    float* temp2;
    half2* gaussiankernel_d;
    int oscillate = 0; //indicate the plane the correct data sits on
    Tiledfp16Gaussian(TestingPlane& plane, float sigma) : plane(plane), sigma(sigma) {
        const int windowsize = 8;
        hipMalloc(&temp, sizeof(float)*plane.width*plane.height);
        hipMalloc(&temp2, sizeof(float)*plane.width*plane.height);
        hipMalloc(&gaussiankernel_d, sizeof(half2)*(4*windowsize+3));
        loadfp16GaussianKernel<<<dim3(1), dim3(2*windowsize+1)>>>(gaussiankernel_d, gaussiankernel_d+2*windowsize+1, windowsize, sigma);
        float* temps[2] = {temp, temp2};
        hipMemcpyDtoD(temps[oscillate], plane.plane_d, sizeof(float)*plane.width*plane.height);
        hipDeviceSynchronize();
    }
    ~Tiledfp16Gaussian(){
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
        const int windowsize = 8;
        float* temps[2] = {temp, temp2};
        float* source = temps[oscillate];
        float* dest = temps[oscillate^1];

        int64_t th_x = 8;
        int64_t th_y = 16;
        int64_t bl_x = (plane.width-1)/(4*th_x)+1;
        int64_t bl_y = (plane.height-1)/(2*th_y)+1;

        Tiledfp16GaussianBlur_Kernel<<<dim3(bl_x*bl_y), dim3(th_x, th_y)>>>(source, dest, plane.width, plane.height, gaussiankernel_d, gaussiankernel_d+2*windowsize+1);

        oscillate ^= 1; //we moved the result in the other place
    }
};