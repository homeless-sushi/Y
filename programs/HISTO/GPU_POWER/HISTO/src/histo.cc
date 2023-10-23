#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <csignal>

#include <boost/program_options.hpp>

#include "Histo/Histo.h"
#include "Histo/HistoCuda.h"
#include "Histo/HistoCpu.h"
#include "Histo/ReadWrite.h"

#include "Knobs/Device.h"

#include "AppRegisterCommon/AppRegister.h"
#include "AppRegisterCommon/Semaphore.h"

#include "AppRegisterClient/AppRegister.h"
#include "AppRegisterClient/Utils.h"

namespace po = boost::program_options;

po::options_description SetupOptions();
void SetupSignals();
void CastKnobs(
    unsigned int gpuNBlocksExp,
    unsigned int gpuBlockSizeExp,
    Knobs::GpuKnobs::N_BLOCKS& gpuNBlocks,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize
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
        4,
        true);
    int dataSemId = semget(getpid(), 1, 0);

    unsigned int deviceId = 0;
    unsigned int cpuThreads = 1;
    unsigned int gpuNBlocksExp = vm["gpu-n-blocks-exp"].as<unsigned int>();
    unsigned int gpuBlockSizeExp = vm["gpu-block-size-exp"].as<unsigned int>();

    Knobs::DEVICE device;
    Knobs::GpuKnobs::N_BLOCKS gpuNBlocks;
    Knobs::GpuKnobs::BLOCK_SIZE gpuBlockSize;

    CastKnobs(
        gpuNBlocksExp,
        gpuBlockSizeExp,
        gpuNBlocks,
        gpuBlockSize
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
    std::string inputFileURL(vm["input-file"].as<std::string>());
    unsigned imgWidth, imgHeight;
    std::vector<unsigned short> rgb;
    Histo::ReadBinaryDataFile(
        inputFileURL,
        imgWidth, imgHeight,
        rgb
    );

    std::unique_ptr<Histo::Histo> histo(
        device == Knobs::DEVICE::GPU ?
        static_cast<Histo::Histo*>(new Histo::HistoCuda(
            rgb,
            gpuNBlocks, gpuBlockSize
        )) :
        static_cast<Histo::Histo*>(new Histo::HistoCpu(
            rgb,
            cpuThreads
        ))
    );
    stopTime = std::chrono::system_clock::now();
    duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
    if(device == Knobs::DEVICE::GPU){
        Histo::HistoCuda* ptr(dynamic_cast<Histo::HistoCuda*>(histo.get()));
        double dataUploadTime = ptr->getDataUploadTime();
        double windUpTime = duration.count() - dataUploadTime;
        std::cout << "WIND UP,CPU," << windUpTime << "\n";
        std::cout << "UPLOAD,GPU," << dataUploadTime << "\n";
    }else{
        std::cout << "WIND UP,CPU," << duration.count() << "\n";
    }
    //STOP: WIND UP

    while(!stop && !error){

        //START: KERNEL
        startTime = std::chrono::system_clock::now();
        histo->run();
        stopTime = std::chrono::system_clock::now();
        if(device == Knobs::DEVICE::GPU){
            Histo::HistoCuda* ptr(dynamic_cast<Histo::HistoCuda*>(histo.get()));
            std::cout << "KERNEL,GPU," << ptr->getKernelTime() << "\n";
        }else{
            duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
            std::cout << "KERNEL,CPU," << duration.count() << "\n";
        }
        //STOP: KERNEL
    }

    //START: WIND DOWN
    startTime = std::chrono::system_clock::now();
    std::vector<unsigned> res = histo->getResult();
    if(vm.count("output-file")){
        Histo::WriteHistogramFile(
            vm["output-file"].as<std::string>(),
            res
        );
    }
    stopTime = std::chrono::system_clock::now();
    duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
    if(device == Knobs::DEVICE::GPU){
        Histo::HistoCuda* ptr(dynamic_cast<Histo::HistoCuda*>(histo.get()));
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

    std::cout << std::endl;
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
    ("instance-name", po::value<std::string>()->default_value("HISTO"), "name of benchmark instance")
    ("target-throughput", po::value<long double>()->default_value(1.0), "target throughput for the kernel")
    ("gpu-n-blocks-exp", po::value<unsigned int>()->default_value(0), "n blocks exp; n blocks = 8*2^X")
    ("gpu-block-size-exp", po::value<unsigned int>()->default_value(0), "block exp; block size = 32*2^X")
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
    unsigned int gpuNBlocksExp,
    unsigned int gpuBlockSizeExp,
    Knobs::GpuKnobs::N_BLOCKS& gpuNBlocks,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize
)
{
    gpuNBlocks = Knobs::nBlocksfromExponent(gpuNBlocksExp);
    gpuBlockSize = Knobs::blockSizefromExponent(gpuBlockSizeExp);
}
