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

#include "AppRegisterCommon/AppRegister.h"
#include "AppRegisterCommon/Semaphore.h"

#include "AppRegisterClient/AppRegister.h"
#include "AppRegisterClient/Utils.h"

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
    typedef std::result_of<decltype(&std::chrono::system_clock::now)()>::type TimePoint;
    TimePoint startLoop, stopLoop;
    TimePoint startTime, stopTime;
    typedef std::chrono::duration<double, std::milli> Duration;
    Duration duration;

    std::cout << "PHASE,DEVICE,DURATION" << "\n";
 
    //START: SETUP
    startTime = std::chrono::system_clock::now();
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
    stopTime = std::chrono::system_clock::now();
    duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
    std::cout << "SETUP,CPU," << duration.count() << "\n";
    //STOP: SETUP

    //Spinlock
    //START: WAIT REGISTRATION
    startTime = std::chrono::system_clock::now();
    while(true){
        if(isRegistered(data)){
            setTickStartTime(data);
            break;
        }
    }
    stopTime = std::chrono::system_clock::now();
    duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
    std::cout << "WAIT REGISTRATION,CPU," << duration.count() << "\n";
    //STOP: WAIT REGISTRATION

    bool error = false;
    while(!stop && !error){

        //START: LOOP
        startLoop = std::chrono::system_clock::now();

        //Read knobs
        //START: CONTROLLER PULL
        startTime = std::chrono::system_clock::now();
        error = binarySemaphoreWait(dataSemId);
        cpuThreads = getNCpuCores(data);
        device = getUseGpu(data) ? Knobs::DEVICE::GPU : Knobs::DEVICE::CPU;
        error = binarySemaphorePost(dataSemId);
        deviceId = static_cast<unsigned int>(device);
        stopTime = std::chrono::system_clock::now();
        duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
        std::cout << "CONTROLLER PULL,CPU," << duration.count() << "\n";
        //STOP: CONTROLLER PULL

        //START: WIND UP
        startTime = std::chrono::system_clock::now();
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
        stopTime = std::chrono::system_clock::now();
        duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
        if(device == Knobs::DEVICE::GPU){
            Sgemm::SgemmCuda* ptr(dynamic_cast<Sgemm::SgemmCuda*>(sgemm.get()));
            double dataUploadTime = ptr->getDataUploadTime();
            double windUpTime = duration.count() - dataUploadTime;
            std::cout << "WIND UP,CPU," << windUpTime << "\n";
            std::cout << "UPLOAD,GPU," << dataUploadTime << "\n";
        }else{
            std::cout << "WIND UP,CPU," << duration.count() << "\n";
        }
        std::cout << "WIND UP,CPU," << duration.count() << "\n";
        //STOP: WIND UP


        //START: KERNEL
        startTime = std::chrono::system_clock::now();
        sgemm->run();
        stopTime = std::chrono::system_clock::now();
        if(device == Knobs::DEVICE::GPU){
            Sgemm::SgemmCuda* ptr(dynamic_cast<Sgemm::SgemmCuda*>(sgemm.get()));
            std::cout << "KERNEL,GPU," << ptr->getKernelTime() << "\n";
        }else{
            duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
            std::cout << "KERNEL,CPU," << duration.count() << "\n";
        }
        //STOP: KERNEL

        //START: WIND DOWN
        startTime = std::chrono::system_clock::now();
        Sgemm::Matrix res(sgemm->getResult());
        if(vm.count("output-file")){
            Sgemm::WriteMatrixFile(vm["output-file"].as<std::string>(), res);
        }
        stopTime = std::chrono::system_clock::now();
        duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
        if(device == Knobs::DEVICE::GPU){
            Sgemm::SgemmCuda* ptr(dynamic_cast<Sgemm::SgemmCuda*>(sgemm.get()));
            double dataDownloadTime = ptr->getDataDownloadTime();
            double windDownTime = duration.count() - dataDownloadTime;
            std::cout << "DOWNLOAD,GPU," << dataDownloadTime << "\n";
            std::cout << "WIND DOWN,CPU," << windDownTime << "\n";
        }else{
            std::cout << "WIND DOWN,CPU," << duration.count() << "\n";
        }
        //START: WIND DOWN

        //Add tick
        //START: CONTROLLER PUSH
        startTime = std::chrono::system_clock::now();
        autosleep(data, targetThroughput);
        error = binarySemaphoreWait(dataSemId);
        addTick(data, 1);
        error = binarySemaphorePost(dataSemId);
                stopTime = std::chrono::system_clock::now();
        duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
        std::cout << "CONTROLLER PUSH,CPU," << duration.count() << "\n";
        //STOP: CONTROLLER PUSH
        
        //STOP: LOOP
        stopLoop = std::chrono::system_clock::now();
        duration = std::chrono::duration<double, std::milli>((stopLoop - startLoop));
        std::cout << "LOOP,NONE," << duration.count() << "\n";
    }
    
    std::cout << std::endl;
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
        std::cerr << "\n";
        std::cerr << "Received signal: " << signal << "\n";
        std::cerr << "Stopping benchmark" << "\n";
        std::cerr << std::endl;

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