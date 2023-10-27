#include <iostream>
#include <string>
#include <memory>

#include "Sgemm/Matrix.h"
#include "Sgemm/Sgemm.h"
#include "Sgemm/SgemmCpu.h"
#include "Sgemm/SgemmCuda.h"
#include "Sgemm/ReadWrite.h"

#include <Knobs/Device.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

po::options_description SetupOptions();
void CastKnobs(
    unsigned int cpuTileSizeExp,
    unsigned int gpuTileSizeExp,
    unsigned int& cpuTileSize,
    Knobs::GpuKnobs::TILE_SIZE& gpuTileSize
);

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

    unsigned int cpuTileSizeExp = vm["cpu-tile-exp"].as<unsigned>();
    unsigned int gpuTileSizeExp = vm["gpu-tile-exp"].as<unsigned>();
    
    Knobs::DEVICE device = vm["gpu"].as<bool>() ? Knobs::DEVICE::GPU : Knobs::DEVICE::CPU;
    unsigned int cpuTileSize;
    Knobs::GpuKnobs::TILE_SIZE gpuTileSize;

    CastKnobs(
        cpuTileSizeExp,
        gpuTileSizeExp,
        cpuTileSize,
        gpuTileSize
    );

    unsigned int cpuThreads = vm["cpu-threads"].as<unsigned>();;

    std::string inputAUrl(vm["input-A"].as<std::string>());
    std::string inputBUrl(vm["input-B"].as<std::string>());
    std::string inputCUrl(vm["input-C"].as<std::string>());
    Sgemm::Matrix a(Sgemm::ReadMatrixFile(inputAUrl));
    Sgemm::Matrix b(Sgemm::ReadMatrixFile(inputBUrl));
    Sgemm::Matrix c(Sgemm::ReadMatrixFile(inputCUrl));

    std::unique_ptr<Sgemm::Sgemm> sgemm( 
        device == Knobs::DEVICE::GPU ?
        static_cast<Sgemm::Sgemm*>(new Sgemm::SgemmCuda(1,1,a,b,c,gpuTileSize)) :
        static_cast<Sgemm::Sgemm*>(new Sgemm::SgemmCpu(1,1,a,b,c,cpuThreads,cpuTileSize))
    );

    unsigned times = vm["times"].as<unsigned>();
    for(unsigned i = 0; i < times; i++){
        sgemm->run();
    }
    
    Sgemm::Matrix res(sgemm->getResult());
    if(vm.count("output-file")){
        Sgemm::WriteMatrixFile(vm["output-file"].as<std::string>(), res);
    }
    
    return 0;
}

void CastKnobs(
    unsigned int cpuTileSizeExp,
    unsigned int gpuTileSizeExp,
    unsigned int& cpuTileSize,
    Knobs::GpuKnobs::TILE_SIZE& gpuTileSize
)
{
    cpuTileSize = Knobs::GetCpuTileSizeFromExponent(cpuTileSizeExp);
    gpuTileSize = Knobs::GetGpuTileSizeFromExponent(gpuTileSizeExp);
}

po::options_description SetupOptions()
{
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Display help message")
    ("input-A,A", po::value<std::string>(), "input file with matrix A")
    ("input-B,B", po::value<std::string>(), "input file with matrix B")
    ("input-C,C", po::value<std::string>(), "input file with matrix C")
    ("output-file,O", po::value<std::string>(), "ouput file with result matrix")
    ("gpu", po::bool_switch(), "use gpu")
    ("cpu-threads", po::value<unsigned int>()->default_value(1), "number of cpu threads")
    ("cpu-tile-exp", po::value<unsigned int>()->default_value(0), "tile exp; block size = 16*2^X")
    ("gpu-tile-exp", po::value<unsigned int>()->default_value(0), "tile exp; block size = 8*2^X")
    ("times,T", po::value<unsigned int>()->default_value(1), "repeat the kernel T times")
    ;

    return desc;
}
