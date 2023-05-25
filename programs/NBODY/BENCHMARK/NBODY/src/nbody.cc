#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <csignal>

#include <boost/program_options.hpp>

#include "Nbody/Body.h"
#include "Nbody/ReadWrite.h"
#include "Nbody/NbodyCpu.h"
#include "Nbody/NbodyCuda.h"

#include "Knobs/Device.h"
#include "Knobs/Precision.h"

#include "AppRegisterCommon/AppRegister.h"
#include "AppRegisterCommon/Semaphore.h"

#include "AppRegisterClient/AppRegister.h"

#include <margot/margot.hpp>

namespace po = boost::program_options;

po::options_description SetupOptions();
void SetupSignals();
void CastKnobs(
    unsigned int gpuBlockSizeExp,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize
);

bool stop = false;

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

    SetupSignals();

    long double targetThroughput = vm["target-throughput"].as<long double>();
    //Attach to controller
    struct app_data* data = registerAttach(
        vm["instance-name"].as<std::string>().c_str(),
        targetThroughput,
        1,
        false);
    int dataSemId = semget(getpid(), 1, 0);

    margot::init();

    unsigned int deviceId = 0;
    unsigned int gpuBlockSizeExp = 0;
    Knobs::DEVICE device;
    Knobs::GpuKnobs::BLOCK_SIZE gpuBlockSize;

    unsigned int cpuThreads = 1;
    unsigned int precision = 100;

    CastKnobs(
        gpuBlockSizeExp,
        gpuBlockSize
    );

    //Spinlock
    while(true){
        if(isRegistered(data)){
            setTickStartTime(data);
            break;
        }
    }

    bool error = false;
    while(!stop && !error){

        //Read knobs
        error = binarySemaphoreWait(dataSemId);
        cpuThreads = getNCpuCores(data);
        device = getUseGpu(data) ? Knobs::DEVICE::GPU : Knobs::DEVICE::CPU;
        error = binarySemaphorePost(dataSemId);
        deviceId = static_cast<unsigned int>(device);

        if(margot::nbody::update(cpuThreads, deviceId, gpuBlockSizeExp, precision)){
            CastKnobs(
                gpuBlockSizeExp,
                gpuBlockSize
            );
            margot::nbody::context().manager.configuration_applied();
        }

        margot::nbody::start_monitors();

        std::string inputFileURL(vm["input-file"].as<std::string>());
        std::vector<Nbody::Body> bodies;
        float targetSimulationTime;
        float targetTimeStep;
        Nbody::ReadBodyFile(inputFileURL, bodies, targetSimulationTime, targetTimeStep);

        float actualTimeStep = targetTimeStep;
        float approximateSimulationTime = Knobs::GetApproximateSimTime(
            targetSimulationTime, 
            targetTimeStep,
            100
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

        //Add tick
        autosleep(data, targetThroughput);
        error = binarySemaphoreWait(dataSemId);
        addTick(data, 1);
        error = binarySemaphorePost(dataSemId);
    }

    registerDetach(data);
    return 0;
}

po::options_description SetupOptions()
{
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Display help message")
    ("input-file,I", po::value<std::string>(), "input file body file")
    ("output-file,O", po::value<std::string>(), "output file result file")
    ("precision,P", po::value<unsigned int>()->default_value(100), "precision in range 0-100")
    ("instance-name", po::value<std::string>()->default_value("NBODY"), "name of benchmark instance")
    ("target-throughput", po::value<long double>()->default_value(1.0), "target throughput for the kernel")
    ;

    return desc;
}

void SetupSignals()
{
    auto stopBenchmark = [](int signal){
        std::cerr << std::endl;
        std::cerr << "Received signal: " << signal << std::endl;
        std::cerr << "Stopping benchmark" << std::endl;

        stop = true;
    };

    std::signal(SIGINT, stopBenchmark);
    std::signal(SIGTERM, stopBenchmark);
}

void CastKnobs(
    unsigned int gpuBlockSizeExp,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize
)
{
    gpuBlockSize = static_cast<Knobs::GpuKnobs::BLOCK_SIZE>(
        Knobs::GpuKnobs::BLOCK_32 << gpuBlockSizeExp
    );
}
