#include "preprocessor.hpp"
#include "CLI_parser.hpp"
#include "common.hpp"
#include "gaussian_impls/tiledGaussian.hpp"
#include "gaussian_impls/tiled3Gaussian.hpp"
#include "gaussian_impls/tiledfp16Gaussian.hpp"
#include "gaussian_impls/classicalGaussian.hpp"
#include "gaussian_impls/classical2Gaussian.hpp"

enum GaussianTypes{
    ClassicalGauss,
    Classical2Gauss,
    TiledGauss,
    Tiled3Gauss,
    Tiledfp16Gauss,
};

GaussianTypes parseGaussian(std::string inp){
    if (inp == "classic") return ClassicalGauss;
    if (inp == "classic2") return Classical2Gauss;
    if (inp == "tiled") return TiledGauss;
    if (inp == "tiled3") return Tiled3Gauss;
    if (inp == "tiledfp16") return Tiledfp16Gauss;
    std::cout << "failed to parse --gauss option, chose classic by default" << std::endl;
    return ClassicalGauss;
}

int main(int argc, char* argv[]){
    std::vector<std::string> args(argc);
    for (int i = 0; i < argc; i++) args[i] = argv[i];
    int width = 1920;
    int height = 1080;
    int sigma = 2;
    int n = 100;
    bool compare = false;
    std::string strGauss = "classic";

    helper::ArgParser cli_parse;

    cli_parse.add_flag({"-w", "--width"}, &width, "width of the performed gaussianBlur for testing");
    cli_parse.add_flag({"-h", "--height"}, &height, "height of the performed gaussianBlur for testing");
    cli_parse.add_flag({"-n"}, &n, "number of gaussianBlur in serie to test to bench");
    cli_parse.add_flag({"--sigma"}, &sigma, "sigma of gaussian blur to test and bench");
    cli_parse.add_flag({"--compare"}, &compare, "Compare the resulting gaussian to a cpu one");
    cli_parse.add_flag({"-g", "--gauss"}, &strGauss, "Choose the gaussian to bench: classic|classic2|tiled");

    if (cli_parse.parse_cli_args(args)!= 0) return 1;

    GaussianTypes gaussiantype = parseGaussian(strGauss);

    int gpucount = 0;
    if (hipGetDeviceCount(&gpucount) != hipSuccess || gpucount == 0){
        std::cout << "No GPU Found" << std::endl;
        return 1;
    }

    TestingPlane reference(width, height);
    ClassicalGaussian base_blur(reference, sigma, 10); //do it on gpu, it is way faster
    if (compare) for (int i = 0; i < n; i++) base_blur.run();
    base_blur.sync();

    std::pair<double, double> res;
    switch (gaussiantype){
        case ClassicalGauss:
            res = bench<ClassicalGaussian>(reference, sigma, n);
            break;
        case Classical2Gauss:
            res = bench<Classical2Gaussian>(reference, sigma, n);
            break;
        case TiledGauss:
            res = bench<TiledGaussian>(reference, sigma, n);
            break;
        case Tiled3Gauss:
            res = bench<Tiled3Gaussian>(reference, sigma, n);
            break;
        case Tiledfp16Gauss:
            res = bench<Tiledfp16Gaussian>(reference, sigma, n);
            break;
    }
    auto [norm2diff, ntime] = res;
    std::cout << "Tested GaussianBlur got error " << norm2diff << " and took " << ntime/n << " microSeconds per blur" << std::endl;
}