#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <csignal>

#include "Nbody/Body.h"
#include "Nbody/ReadWrite.h"
#include "Nbody/NbodyCpu.h"
#include "Nbody/NbodyCuda.h"

#include "Knobs/Device.h"
#include "Knobs/Precision.h"

#include <boost/program_options.hpp>

#include <margot/margot.hpp>

#include "AppRegisterServer/Frequency.h"

#include "AppRegisterCommon/AppRegister.h"
#include "AppRegisterCommon/Semaphore.h"

#include "AppRegisterClient/AppRegister.h"
#include "AppRegisterClient/Utils.h"

namespace po = boost::program_options;

po::options_description SetupOptions();
void SetupSignals();
unsigned int extractSize(const std::string& inputUrl);
void CastKnobs(
    unsigned int gpuBlockSizeExp,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize
);

bool stop = false;

int main(int argc, char *argv[])
{
 
    //START: SETUP
    po::options_description desc(SetupOptions());
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 0;
    }

    SetupSignals();

    margot::init();

    long double targetThroughput = vm["target-throughput"].as<long double>();
    unsigned int target_precision = vm["target-precision"].as<unsigned int>();
    unsigned int inputSize = extractSize(vm["input-file"].as<std::string>());
    struct app_data* data = registerAttach(
        vm["instance-name"].as<std::string>().c_str(),
        "NBODY", inputSize,
        targetThroughput,
        4,
        true,
        true, target_precision
    );
    int dataSemId = semget(getpid(), 1, 0);

    Frequency::CPU_FRQ cpuFreq = Frequency::getMinCpuFreq();
    Frequency::GPU_FRQ gpuFreq = Frequency::getMinGpuFreq();
    unsigned int useGpu = 0;
    unsigned int nCores = 1;
    unsigned int gpuBlockSizeExp = 0;
    Knobs::GpuKnobs::BLOCK_SIZE gpuBlockSize;
    unsigned int precision = 0;

    CastKnobs(
        gpuBlockSizeExp,
        gpuBlockSize
    );
    //STOP: SETUP

    //Spinlock
    //START: WAIT REGISTRATION
    while(true){
        if(isRegistered(data)){
            setTickStartTime(data);
            break;
        }
    }
    //STOP: WAIT REGISTRATION

    bool error = false;
    while(!stop && !error){

        //START: LOOP

        //Read knobs
        //START: CONTROLLER PULL
        error = binarySemaphoreWait(dataSemId);
        nCores = getNCpuCores(data);
        useGpu = getUseGpu(data) ? Knobs::DEVICE::GPU : Knobs::DEVICE::CPU;
        cpuFreq = static_cast<Frequency::CPU_FRQ>(getCpuFreq(data));
        gpuFreq = static_cast<Frequency::GPU_FRQ>(getGpuFreq(data));
        error = binarySemaphorePost(dataSemId);
        //STOP: CONTROLLER PULL

        //START: MARGOT PULL
        if(margot::nbody::update(useGpu, cpuFreq, gpuFreq, inputSize, nCores, gpuBlockSizeExp, precision)){   
            CastKnobs(
                gpuBlockSizeExp,
                gpuBlockSize
            );
            margot::nbody::context().manager.configuration_applied(); 
        }
        margot::nbody::start_monitors();
        //STOP: MARGOT PULL

        //START: WIND UP
        std::vector<Nbody::Body> bodies;
        float targetSimulationTime;
        float targetTimeStep;
        std::string inputFileURL(vm["input-file"].as<std::string>());
        Nbody::ReadBodyFile(inputFileURL, bodies, targetSimulationTime, targetTimeStep);

        float actualTimeStep = targetTimeStep;
        float approximateSimulationTime = Knobs::GetApproximateSimTime(
            targetSimulationTime, 
            targetTimeStep,
            precision
        );

        std::unique_ptr<Nbody::Nbody> nbody( 
            useGpu == Knobs::DEVICE::GPU ?
            static_cast<Nbody::Nbody*>(new NbodyCuda::NbodyCuda(bodies, approximateSimulationTime, actualTimeStep, gpuBlockSize)) :
            static_cast<Nbody::Nbody*>(new NbodyCpu::NbodyCpu(bodies, approximateSimulationTime, actualTimeStep, nCores))
        );
        //STOP: WIND UP

        //START: KERNEL
        nbody->run();
        //STOP: KERNEL
        
        //START: WIND DOWN
        bodies = nbody->getResult();
        if(vm.count("output-file")){
            Nbody::WriteBodyFile(vm["output-file"].as<std::string>(), 
                bodies,
                nbody->getSimulatedTime(),
                actualTimeStep
            );
        }
        //START: WIND DOWN

        //START: MARGOT PUSH
        margot::nbody::stop_monitors();
        margot::nbody::push_custom_monitor_values(precision);
        margot::nbody::log();
        //STOP: MARGOT PUSH

        //Add tick
        //START: CONTROLLER PUSH
        autosleep(data, targetThroughput);
        error = binarySemaphoreWait(dataSemId);
        addTick(data, 1);
        setCurrPrecision(data, precision);
        error = binarySemaphorePost(dataSemId);
        //STOP: CONTROLLER PUSH
        
        //STOP: LOOP
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
    ("instance-name", po::value<std::string>()->default_value("NBODY"), "name of benchmark instance")
    ("target-throughput", po::value<long double>()->default_value(1.0), "target throughput for the kernel")
    ("target-precision,P", po::value<unsigned int>()->default_value(100), "target precision in range 0-100")
    ;

    return desc;
}

void SetupSignals()
{
    auto stopBenchmark = [](int signal){
        std::cerr << "\n";
        std::cerr << "Received signal: " << signal << "\n";
        std::cerr << "Stopping benchmark" << "\n";
        std::cerr << std::endl;

        stop = true;
    };

    std::signal(SIGINT, stopBenchmark);
    std::signal(SIGTERM, stopBenchmark);
}

unsigned int extractSize(const std::string& inputUrl) {
    unsigned int size = 0;
    bool started = false;
    int nDigit = 1;

    for (auto rit = inputUrl.rbegin(); rit != inputUrl.rend(); ++rit) {
        const auto c = *rit; 
        if (std::isdigit(c)) {
            started = true;
            size += (c - '0') * nDigit;
            nDigit *= 10;
        } else if (started) {
            break;
        }
    }

    return size;
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
