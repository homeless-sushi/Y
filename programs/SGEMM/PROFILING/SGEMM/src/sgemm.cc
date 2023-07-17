#include <iostream>
#include <string>
#include <memory>

#include <csignal>

#include "Sgemm/Matrix.h"
#include "Sgemm/Sgemm.h"
#include "Sgemm/SgemmCpu.h"
#include "Sgemm/SgemmCuda.h"
#include "Sgemm/ReadWrite.h"

#include <Knobs/Device.h>

#include <boost/program_options.hpp>

#include <margot/margot.hpp>

namespace po = boost::program_options;

po::options_description SetupOptions();
void CastKnobs(
    unsigned int deviceId,
    unsigned int cpuTileSizeExp,
    unsigned int gpuTileSizeExp,
    Knobs::DEVICE& device,
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

    margot::init();
    margot::sgemm::context().manager.wait_for_knowledge(10);

    unsigned int deviceId = 0;
    unsigned int cpuTileSizeExp = 0;
    unsigned int gpuTileSizeExp = 0;
    
    Knobs::DEVICE device;
    unsigned int cpuTileSize;
    Knobs::GpuKnobs::TILE_SIZE gpuTileSize;

    unsigned int cpuThreads = 1;

    CastKnobs(
        deviceId,
        cpuTileSizeExp,
        gpuTileSizeExp,
        device,
        cpuTileSize,
        gpuTileSize
    );

    while(margot::sgemm::context().manager.in_design_space_exploration()){

        if(margot::sgemm::update(cpuThreads, cpuTileSizeExp, deviceId, gpuTileSizeExp)){
            CastKnobs(
                deviceId,
                cpuTileSizeExp,
                gpuTileSizeExp,
                device,
                cpuTileSize,
                gpuTileSize
            );
            margot::sgemm::context().manager.configuration_applied();
        }

        std::string inputAUrl(vm["input-file"].as<std::string>());
        std::string inputBUrl(vm["input-file"].as<std::string>());
        std::string inputCUrl(vm["input-file"].as<std::string>());
        Sgemm::Matrix a(Sgemm::ReadMatrixFile(inputAUrl));
        Sgemm::Matrix b(Sgemm::ReadMatrixFile(inputBUrl));
        Sgemm::Matrix c(Sgemm::ReadMatrixFile(inputCUrl));

        std::unique_ptr<Sgemm::Sgemm> sgemm( 
                    device == Knobs::DEVICE::GPU ?
                    static_cast<Sgemm::Sgemm*>(new Sgemm::SgemmCuda(1,1,a,b,c,gpuTileSize)) :
                    static_cast<Sgemm::Sgemm*>(new Sgemm::SgemmCpu(1,1,a,b,c,cpuThreads,cpuTileSize))
                );
        sgemm->run();
        Sgemm::Matrix res(sgemm->getResult());

        if(vm.count("output-file")){
            Sgemm::WriteMatrixFile(vm["output-file"].as<std::string>(), res);
        }

        margot::sgemm::stop_monitors();
        margot::sgemm::push_custom_monitor_values();
        margot::sgemm::log();
    }
    
    return 0;
}

po::options_description SetupOptions()
{
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Display help message")
    ("input-file,I", po::value<std::string>(), "input file with matrices A,B,C")
    ("output-file,O", po::value<std::string>(), "ouput file with result matrix")
    ;

    return desc;
}

void CastKnobs(
    unsigned int deviceId,
    unsigned int cpuTileSizeExp,
    unsigned int gpuTileSizeExp,
    Knobs::DEVICE& device,
    unsigned int& cpuTileSize,
    Knobs::GpuKnobs::TILE_SIZE& gpuTileSize
)
{
    device = static_cast<Knobs::DEVICE>(deviceId);
    cpuTileSize = 16 << cpuTileSizeExp;
    gpuTileSize = static_cast<Knobs::GpuKnobs::TILE_SIZE>(
        Knobs::GpuKnobs::TILE_8 << gpuTileSizeExp
    );
}
