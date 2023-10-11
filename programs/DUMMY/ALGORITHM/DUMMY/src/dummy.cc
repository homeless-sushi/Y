#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <boost/program_options.hpp>

#include "Dummy/Dummy.h"
#include "Dummy/ReadWrite.h"

#include "CudaError/CudaError.h"

#define SM_COUNT = 1
#define SP_COUNT = 128
#define WARP_SIZE = 32

namespace po = boost::program_options;

po::options_description SetupOptions();

int main(int argc, char *argv[])
{
    po::options_description desc(SetupOptions());
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 0;
    }

    auto startWindUp = std::chrono::system_clock::now();
    std::vector<float> fileData;
    std::string inputFileURL(vm["input-file"].as<std::string>());
    Dummy::ReadFile(inputFileURL, fileData);

    cudaDeviceProp deviceProp;
    CudaErrorCheck(cudaGetDeviceProperties(&deviceProp, 0));
    unsigned int smCount = SM_COUNT;//deviceProp.multiProcessorCount;
    unsigned int warpSize = WARP_SIZE;//deviceProp.warpSize;
    unsigned int nThreads = SP_COUNT;
    std::vector<float> dummyData(nThreads);
    std::copy(fileData.begin(), fileData.begin() + nThreads, dummyData.begin());

    Dummy::Dummy dummy(dummyData, smCount, warpSize, vm["times"].as<unsigned int>());
    auto endWindUp = std::chrono::system_clock::now();
    std::cout 
        << std::chrono::duration_cast<std::chrono::milliseconds>(endWindUp - startWindUp).count() 
        << ",";

    auto startKernel = std::chrono::system_clock::now();
    dummy.run();
    dummyData = dummy.getResult();
    auto endKernel = std::chrono::system_clock::now();
    std::cout 
        << std::chrono::duration_cast<std::chrono::milliseconds>(endKernel - startKernel).count() 
        << ",";

    auto startWindDown = std::chrono::system_clock::now();
    if(vm.count("output-file")){
        std::string outputFileURL(vm["output-file"].as<std::string>());
        Dummy::WriteFile(outputFileURL, dummyData);
    }
    auto endWindDown = std::chrono::system_clock::now();
    std::cout
        << std::chrono::duration_cast<std::chrono::milliseconds>(endWindDown - startWindDown).count();

    
    return 0;
}

po::options_description SetupOptions()
{
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Display help message")
    ("input-file,I", po::value<std::string>(), "input file body file")
    ("output-file,O", po::value<std::string>(), "output file result file")
    ("times,T", po::value<unsigned int>()->default_value(1), "knob for CUDA kernel duration")
    ;

    return desc;
}
