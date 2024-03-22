#include <iostream>
#include <chrono>
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
    unsigned int cpuTileSizeExp,
    unsigned int gpuTileSizeExp,
    unsigned int& cpuTileSize,
    Knobs::GpuKnobs::TILE_SIZE& gpuTileSize
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
    unsigned int inputSize = extractSize(vm["input-file"].as<std::string>());
    //Attach to controller
    struct app_data* data = registerAttach(
        vm["instance-name"].as<std::string>().c_str(),
        "SGEMM", inputSize,
        targetThroughput,
        4,
        true);
    int dataSemId = semget(getpid(), 1, 0);

    Frequency::CPU_FRQ cpuFreq = Frequency::getMinCpuFreq();
    Frequency::GPU_FRQ gpuFreq = Frequency::getMinGpuFreq();
    unsigned int useGpu = 0;
    unsigned int nCores = 1;
    unsigned int cpuTileSizeExp = 0;
    unsigned int cpuTileSize;
    unsigned int gpuTileSizeExp = 0;
    Knobs::GpuKnobs::TILE_SIZE gpuTileSize;

    CastKnobs(
        cpuTileSizeExp,
        gpuTileSizeExp,
        cpuTileSize,
        gpuTileSize
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
        if(margot::sgemm::update(useGpu, cpuFreq, gpuFreq, inputSize, nCores, cpuTileSizeExp, gpuTileSizeExp)){   
            CastKnobs(
                cpuTileSizeExp,
                gpuTileSizeExp,
                cpuTileSize,
                gpuTileSize
            );
            margot::sgemm::context().manager.configuration_applied(); 
        }
        margot::sgemm::start_monitors();
        //STOP: MARGOT PULL

        //START: WIND UP
        std::string inputAUrl(vm["input-file"].as<std::string>());
        std::string inputBUrl(vm["input-file"].as<std::string>());
        std::string inputCUrl(vm["input-file"].as<std::string>());
        Sgemm::Matrix a(Sgemm::ReadMatrixFile(inputAUrl));
        Sgemm::Matrix b(Sgemm::ReadMatrixFile(inputBUrl));
        Sgemm::Matrix c(Sgemm::ReadMatrixFile(inputCUrl));

        std::unique_ptr<Sgemm::Sgemm> sgemm( 
            useGpu ?
            static_cast<Sgemm::Sgemm*>(new Sgemm::SgemmCuda(1,1,a,b,c,gpuTileSize)) :
            static_cast<Sgemm::Sgemm*>(new Sgemm::SgemmCpu(1,1,a,b,c,nCores,cpuTileSize))
        );
        //STOP: WIND UP


        //START: KERNEL
        unsigned times = vm["times"].as<unsigned>();
        for(unsigned i = 0; i < times; i++){
            sgemm->run();
        }
        //STOP: KERNEL

        //START: WIND DOWN
        Sgemm::Matrix res(sgemm->getResult());
        if(vm.count("output-file")){
            Sgemm::WriteMatrixFile(vm["output-file"].as<std::string>(), res);
        }
        //START: WIND DOWN

        //START: MARGOT PUSH
        margot::sgemm::stop_monitors();
        margot::sgemm::log();
        //STOP: MARGOT PUSH

        //Add tick
        //START: CONTROLLER PUSH
        autosleep(data, targetThroughput);
        error = binarySemaphoreWait(dataSemId);
        addTick(data, 1);
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
    ("input-file,I", po::value<std::string>(), "input file with matrices A,B,C")
    ("output-file,O", po::value<std::string>(), "ouput file with result matrix")
    ("instance-name", po::value<std::string>()->default_value("SGEMM"), "name of benchmark instance")
    ("target-throughput", po::value<long double>()->default_value(1.0), "target throughput for the kernel")
    ("times,T", po::value<unsigned int>()->default_value(1), "repeat the kernel T times")
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
    unsigned int cpuTileSizeExp,
    unsigned int gpuTileSizeExp,
    unsigned int& cpuTileSize,
    Knobs::GpuKnobs::TILE_SIZE& gpuTileSize
)
{
    cpuTileSize = Knobs::GetCpuTileSizeFromExponent(cpuTileSizeExp);
    gpuTileSize = Knobs::GetGpuTileSizeFromExponent(gpuTileSizeExp);
}
