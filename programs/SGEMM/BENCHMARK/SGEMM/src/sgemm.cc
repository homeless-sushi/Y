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

#include "AppRegisterCommon/AppRegister.h"
#include "AppRegisterCommon/Semaphore.h"

#include "AppRegisterClient/AppRegister.h"
#include "AppRegisterClient/Utils.h"

#include <margot/margot.hpp>

namespace po = boost::program_options;

po::options_description SetupOptions();
void SetupSignals();
void CastKnobs(
    unsigned int cpuTileSizeExp,
    unsigned int gpuTileSizeExp,
    unsigned int& cpuTileSize,
    Knobs::GpuKnobs::TILE_SIZE& gpuTileSize
);

bool stop = false;

int main(int argc, char *argv[])
{
    std::cout << "EVENT,TYPE,DEVICE,TIMESTAMP" << std::endl;
 
    //START: SETUP
    std::cout << "SETUP,START,CPU," << now() << std::endl;
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
    unsigned int cpuTileSizeExp = 0;
    unsigned int gpuTileSizeExp = 0;
    
    Knobs::DEVICE device;
    unsigned int cpuTileSize;
    Knobs::GpuKnobs::TILE_SIZE gpuTileSize;

    unsigned int cpuThreads = 1;

    CastKnobs(
        cpuTileSizeExp,
        gpuTileSizeExp,
        cpuTileSize,
        gpuTileSize
    );
    std::cout << "SETUP,STOP,CPU," << now() << std::endl;
    //STOP: SETUP

    //Spinlock
    //START: WAIT REGISTRATION
    std::cout << "WAIT REGISTRATION,START,CPU," << now() << std::endl;
    while(true){
        if(isRegistered(data)){
            setTickStartTime(data);
            break;
        }
    }
    std::cout << "WAIT REGISTRATION,STOP,CPU," << now() << std::endl;
    //STOP: WAIT REGISTRATION

    bool error = false;
    while(!stop && !error){

        //Read knobs
        //START: CONTROLLER PULL
        std::cout << "CONTROLLER PULL,START,CPU," << now() << std::endl;
        error = binarySemaphoreWait(dataSemId);
        cpuThreads = getNCpuCores(data);
        device = getUseGpu(data) ? Knobs::DEVICE::GPU : Knobs::DEVICE::CPU;
        error = binarySemaphorePost(dataSemId);
        deviceId = static_cast<unsigned int>(device);
        std::cout << "CONTROLLER PULL,STOP,CPU," << now() << std::endl;
        //STOP: CONTROLLER PULL

        //START: MARGOT PULL
        std::cout << "MARGOT PULL,START,CPU," << now() << std::endl;
        if(margot::sgemm::update(cpuThreads, cpuTileSizeExp, deviceId, gpuTileSizeExp)){
            CastKnobs(
                cpuTileSizeExp,
                gpuTileSizeExp,
                cpuTileSize,
                gpuTileSize
            );
            margot::sgemm::context().manager.configuration_applied();
        }
        std::cout << "MARGOT PULL,STOP,CPU," << now() << std::endl;
        //STOP: MARGOT PULL

        //START: WIND UP
        std::cout << "WIND UP,START,CPU," << now() << std::endl;
        margot::sgemm::start_monitors();
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
        std::cout << "WIND UP,STOP,CPU," << now() << std::endl;
        //STOP: WIND UP

        //START: KERNEL
        std::cout << "KERNEL,START," << Knobs::DeviceToString(device) << "," << now() << std::endl;
        sgemm->run();
        Sgemm::Matrix res(sgemm->getResult());
        std::cout << "KERNEL,STOP," << Knobs::DeviceToString(device) << "," << now() << std::endl;
        //STOP: KERNEL

        //START: WIND DOWN
        std::cout << "WIND DOWN,START,CPU," << now() << std::endl;
        if(vm.count("output-file")){
            Sgemm::WriteMatrixFile(vm["output-file"].as<std::string>(), res);
        }
        std::cout << "WIND DOWN,STOP,CPU," << now() << std::endl;
        //START: WIND DOWN

        //START: MARGOT PUSH
        std::cout << "MARGOT PUSH,START,CPU," << now() << std::endl;
        margot::sgemm::stop_monitors();
        margot::sgemm::push_custom_monitor_values();
        std::cout << "MARGOT PUSH,STOP,CPU," << now() << std::endl;
        //STOP: MARGOT PUSH

        //Add tick
        //START: CONTROLLER PUSH
        std::cout << "CONTROLLER PUSH,START,CPU," << now() << std::endl;
        autosleep(data, targetThroughput);
        error = binarySemaphoreWait(dataSemId);
        addTick(data, 1);
        error = binarySemaphorePost(dataSemId);
        std::cout << "CONTROLLER PUSH,STOP,CPU," << now() << std::endl;
        //STOP: CONTROLLER PUSH
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
    unsigned int cpuTileSizeExp,
    unsigned int gpuTileSizeExp,
    unsigned int& cpuTileSize,
    Knobs::GpuKnobs::TILE_SIZE& gpuTileSize
)
{
    cpuTileSize = 16 << cpuTileSizeExp;
    gpuTileSize = static_cast<Knobs::GpuKnobs::TILE_SIZE>(
        Knobs::GpuKnobs::TILE_8 << gpuTileSizeExp
    );
}
