#pragma once

struct TestingPlane{
    float* plane_d;
    float* plane;
    const int width;
    const int height;
    TestingPlane(int width, int height) : width(width), height(height) {
        plane = (float*)malloc(sizeof(float)*width*height);
        hipMalloc(&plane_d, sizeof(float)*width*height);

        //draw a checkboard pattern
        for (int y = 0; y < height; y++){
            for (int x = 0; x < width; x++){
                plane[y*width + x] = 1+((y+x)%2);
            }
        }
        hipMemcpyHtoD(plane_d, plane, sizeof(float)*width*height);
    }
    void copyGPU() const {
        hipMemcpyHtoD(plane_d, plane, sizeof(float)*width*height);
    }
    void copyCPU() const {
        hipMemcpyDtoH(plane, plane_d, sizeof(float)*width*height);
    }
    double getNorm2Diff(const TestingPlane& other){
        //will get on CPU because it is simpler for now
        copyCPU();
        other.copyCPU();
        double accum = 0;
        for (int i = 0; i < width*height; i++){
            accum += (plane[i] - other.plane[i])*(plane[i] - other.plane[i]);
        }
        return std::sqrt(accum/width/height);
    }
    void cpu_blur(float sigma){
        const int window = sigma * 10;
        float* gaussianprecalc = (float*)malloc(sizeof(float)*(2*window+1));
        for (int i = 0; i <= 2*window; i++){
            gaussianprecalc[i] = std::exp(-(window-i)*(window-i)/(2*sigma*sigma))/(std::sqrt(TAU*sigma*sigma));
        }
        CPU_HorizontalBlur(sigma, gaussianprecalc);
        CPU_VerticalBlur(sigma, gaussianprecalc);
        free(gaussianprecalc);
        copyGPU();
    }
    ~TestingPlane(){
        free(plane);
        hipFree(plane_d);
    }
private:
    void CPU_HorizontalBlur(float sigma, float* gaussianprecalc){
        const int window = sigma * 10;
        float* temp = (float*)malloc(sizeof(float)*width*height);
        for (int y = 0; y < height; y++){
            for (int x = 0; x < width; x++){
                float tot = 0;
                float acc = 0;
                for (int i = -window; i <= window; i++){
                    if (x+i >= 0 && x+i < width) {
                        tot += gaussianprecalc[i+window];
                        acc += gaussianprecalc[i+window]*plane[y*width+(x+i)];
                    }
                }
                temp[y*width+x] = acc/tot;
            }
        }
        memcpy(plane, temp, sizeof(float)*width*height);
        free(temp);
    }
    void CPU_VerticalBlur(float sigma, float* gaussianprecalc){
        const int window = sigma * 10;
        float* temp = (float*)malloc(sizeof(float)*width*height);
        for (int y = 0; y < height; y++){
            for (int x = 0; x < width; x++){
                float tot = 0;
                float acc = 0;
                for (int i = -window; i <= window; i++){
                    if (y+i >= 0 && y+i < height) {
                        tot += gaussianprecalc[i+window];
                        acc += gaussianprecalc[i+window]*plane[(y+i)*width+x];
                    }
                }
                temp[y*width+x] = acc/tot;
            }
        }
        memcpy(plane, temp, sizeof(float)*width*height);
        free(temp);
    }
};

//templated by a fonctor that takes a testingPlane as argument and run a type of gaussian blur
template <typename FONCTOR>
std::pair<double, double> bench(TestingPlane& reference, float sigma, int n){
    TestingPlane plane(reference.width, reference.height);
    FONCTOR gaussianimp(plane, sigma);

    auto init = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++){
        gaussianimp.run();
    }

    gaussianimp.sync();

    auto fin = std::chrono::high_resolution_clock::now();
    double micro = std::chrono::duration_cast<std::chrono::microseconds>(fin - init).count();
    return std::make_pair(reference.getNorm2Diff(plane), micro);
}