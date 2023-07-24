#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "Nbody/Body.h"
#include "Nbody/ReadWrite.h"
#include "Nbody/NbodyCpu.h"
#include "Nbody/NbodyCuda.h"

#include "Knobs/Device.h"
#include "Knobs/Precision.h"

#include <Dvfs/Dvfs.h>

#include <boost/program_options.hpp>

#include <margot/margot.hpp>

namespace po = boost::program_options;

po::options_description SetupOptions();
void CastKnobs(
    unsigned int deviceId,
    unsigned int gpuBlockSizeExp,
    unsigned int inputSize,
    Knobs::DEVICE& device,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize,
    std::string& inputUrl
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
    margot::nbody::context().manager.wait_for_knowledge(10);

    unsigned int deviceId = 0;
    unsigned int gpuBlockSizeExp = 0;
    unsigned int inputSize = 512;
    
    Knobs::DEVICE device;
    Knobs::GpuKnobs::BLOCK_SIZE gpuBlockSize;
    std::string inputUrl;

    unsigned int cpuFreq = Dvfs::CPU_FRQ::FRQ_102000KHz;
    unsigned int gpuFreq = Dvfs::GPU_FRQ::FRQ_76800000Hz;
    unsigned int cpuThreads = 1;
    unsigned int precision = 1;

    CastKnobs(
        deviceId,
        gpuBlockSizeExp,
        inputSize,
        device,
        gpuBlockSize,
        inputUrl
    );

    while(margot::nbody::context().manager.in_design_space_exploration()){

        if(margot::nbody::update(
            cpuFreq,
            cpuThreads,
            deviceId,
            gpuBlockSizeExp,
            gpuFreq,
            inputSize,
            precision))
        {
            CastKnobs(
                deviceId,
                gpuBlockSizeExp,
                inputSize,
                device,
                gpuBlockSize,
                inputUrl
            );
            margot::nbody::context().manager.configuration_applied();
        }

        Dvfs::SetCpuFreq(static_cast<Dvfs::CPU_FRQ>(cpuFreq));
        Dvfs::SetGpuFreq(static_cast<Dvfs::GPU_FRQ>(gpuFreq));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        margot::nbody::start_monitors();

        std::vector<Nbody::Body> bodies;
        float targetSimulationTime;
        float targetTimeStep;
        Nbody::ReadBodyFile(inputUrl, bodies, targetSimulationTime, targetTimeStep);

        float actualTimeStep = targetTimeStep;
        float approximateSimulationTime = Knobs::GetApproximateSimTime(
            targetSimulationTime, 
            targetTimeStep,
            precision
        );

        std::unique_ptr<Nbody::Nbody> nbody( 
            device == Knobs::DEVICE::GPU ?
            static_cast<Nbody::Nbody*>(new NbodyCuda::NbodyCuda(bodies, actualTimeStep, gpuBlockSize)) :
            static_cast<Nbody::Nbody*>(new NbodyCpu::NbodyCpu(bodies, actualTimeStep, cpuThreads))
        );
        float actualSimulationTime;
        for(actualSimulationTime = 0.f; actualSimulationTime < targetSimulationTime; actualSimulationTime+=actualTimeStep){
            nbody->run();
        }
        bodies = nbody->getResult();
        
        if(vm.count("output-file")){
            Nbody::WriteBodyFile(vm["output-file"].as<std::string>(), 
                bodies,
                actualSimulationTime,
                actualTimeStep
            );
        }

        margot::nbody::stop_monitors();
        margot::nbody::push_custom_monitor_values();
        margot::nbody::log();
    }

    return 0;
}

po::options_description SetupOptions()
{
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Display help message")
    ("output-file,O", po::value<std::string>(), "output file result file")
    ;

    return desc;
}

void CastKnobs(
    unsigned int deviceId,
    unsigned int gpuBlockSizeExp,
    unsigned int inputSize,
    Knobs::DEVICE& device,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize,
    std::string& inputUrl
)
{
    device = static_cast<Knobs::DEVICE>(deviceId);
    gpuBlockSize = static_cast<Knobs::GpuKnobs::BLOCK_SIZE>(
        Knobs::GpuKnobs::BLOCK_32 << gpuBlockSizeExp
    );
    std::stringstream inputUrlStream;
    inputUrlStream << "/home/miele/Vivian/Thesis/apps/Y/programs/NBODY/data/in/" << inputSize << ".txt";
    inputUrl = inputUrlStream.str();
}
