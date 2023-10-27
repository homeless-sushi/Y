#include <chrono>
#include <iostream>
#include <string>
#include <memory>

#include <csignal>

#include "Graph/Graph.h"
#include "Graph/ReadWrite.h"

#include "Bfs/Bfs.h"
#include "Bfs/BfsCpu.h"
#include "Bfs/BfsCuda.h"
#include "Bfs/ReadWrite.h"

#include "Knobs/Device.h"

#include <boost/program_options.hpp>

#include "AppRegisterCommon/AppRegister.h"
#include "AppRegisterCommon/Semaphore.h"

#include "AppRegisterClient/AppRegister.h"
#include "AppRegisterClient/Utils.h"

namespace po = boost::program_options;

po::options_description SetupOptions();
void SetupSignals();
void CastKnobs(
    unsigned int deviceId,
    unsigned int gpuBlockSizeExp,
    unsigned int gpuChunkFactorExp,
    bool useGpuEdgesTextureMem,
    bool useGpuOffsetsTextureMem,
    Knobs::DEVICE& device,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize,
    Knobs::GpuKnobs::CHUNK_FACTOR& gpuChunckFactor,
    Knobs::GpuKnobs::MEMORY_TYPE& edgesOffsetsMemType,
    Knobs::GpuKnobs::MEMORY_TYPE& edgesMemType
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
    unsigned int gpuBlockSizeExp = vm["gpu-block-exp"].as<unsigned int>();
    unsigned int gpuChunkFactorExp = vm["gpu-chunk-exp"].as<unsigned int>();
    bool useGpuEdgesTextureMem = vm["gpu-edges-texture-mem"].as<bool>();
    bool useGpuOffsetsTextureMem = vm["gpu-offsets-texture-mem"].as<bool>();
    
    Knobs::DEVICE device;
    Knobs::GpuKnobs::BLOCK_SIZE gpuBlockSize;
    Knobs::GpuKnobs::CHUNK_FACTOR gpuChunckFactor;
    Knobs::GpuKnobs::MEMORY_TYPE edgesOffsetsMemType;
    Knobs::GpuKnobs::MEMORY_TYPE edgesMemType;

    unsigned int cpuThreads = 1;

    CastKnobs(
        deviceId,
        gpuBlockSizeExp,
        gpuChunkFactorExp,
        useGpuEdgesTextureMem,
        useGpuOffsetsTextureMem,
        device,
        gpuBlockSize,
        gpuChunckFactor,
        edgesOffsetsMemType,
        edgesMemType
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
        std::string inputFileURL(vm["input-file"].as<std::string>());
        Graph::Graph graph(Graph::ReadGraphFile(inputFileURL));

        std::unique_ptr<Bfs::Bfs> bfs( 
            (device == Knobs::DEVICE::GPU) ?
            static_cast<Bfs::Bfs*>(new Bfs::BfsCuda(
                graph,
                0,
                static_cast<unsigned int>(gpuBlockSize),
                static_cast<unsigned int>(gpuChunckFactor),
                static_cast<bool>(edgesOffsetsMemType),
                static_cast<bool>(edgesMemType)
            )) :
            static_cast<Bfs::Bfs*>(new Bfs::BfsCpu(
                graph,
                0,
                cpuThreads
            ))
        );
        stopTime = std::chrono::system_clock::now();
        duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
        if(device == Knobs::DEVICE::GPU){
            Bfs::BfsCuda* ptr(dynamic_cast<Bfs::BfsCuda*>(bfs.get()));
            double dataUploadTime = ptr->getDataUploadTime();
            double windUpTime = duration.count() - dataUploadTime;
            std::cout << "WIND UP,CPU," << windUpTime << "\n";
            std::cout << "UPLOAD,GPU," << dataUploadTime << "\n";
        }else{
            std::cout << "WIND UP,CPU," << duration.count() << "\n";
        }
        //STOP: WIND UP

        //START: KERNEL
        startTime = std::chrono::system_clock::now();
        bfs->run();
        stopTime = std::chrono::system_clock::now();
        if(device == Knobs::DEVICE::GPU){
            Bfs::BfsCuda* ptr(dynamic_cast<Bfs::BfsCuda*>(bfs.get()));
            std::cout << "KERNEL,GPU," << ptr->getKernelTime() << "\n";
        }else{
            duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
            std::cout << "KERNEL,CPU," << duration.count() << "\n";
        }
        //STOP: KERNEL

        //START: WIND DOWN
        startTime = std::chrono::system_clock::now();
        std::vector<int> costs = bfs->getResult();
        if(vm.count("output-file")){
            Bfs::WriteCosts(vm["output-file"].as<std::string>(), costs);
        }
        stopTime = std::chrono::system_clock::now();
        duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
        if(device == Knobs::DEVICE::GPU){
            Bfs::BfsCuda* ptr(dynamic_cast<Bfs::BfsCuda*>(bfs.get()));
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

    ("input-file,I", po::value<std::string>(), "input atoms file")
    ("output-file,O", po::value<std::string>(), "output lattice result file")

    ("instance-name", po::value<std::string>()->default_value("BFS"), "name of benchmark instance")
    ("target-throughput", po::value<long double>()->default_value(1.0), "target throughput for the kernel")

    ("gpu-block-exp", po::value<unsigned int>()->default_value(0), "block exp; block size = 32*2^X")
    ("gpu-chunk-exp", po::value<unsigned int>()->default_value(0), "chunk exp; chunk size = 1*2^X")
    ("gpu-edges-texture-mem", po::bool_switch(), "use gpu texture memory for edges")
    ("gpu-offsets-texture-mem", po::bool_switch(), "use gpu texture memory for offsets")
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
    unsigned int deviceId,
    unsigned int gpuBlockSizeExp,
    unsigned int gpuChunkFactorExp,
    bool useGpuEdgesTextureMem,
    bool useGpuOffsetsTextureMem,
    Knobs::DEVICE& device,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize,
    Knobs::GpuKnobs::CHUNK_FACTOR& gpuChunckFactor,
    Knobs::GpuKnobs::MEMORY_TYPE& gpuOffsetsTextureMem,
    Knobs::GpuKnobs::MEMORY_TYPE& gpuEdgesTextureMem
){
    device = static_cast<Knobs::DEVICE>(deviceId);
    gpuBlockSize = static_cast<Knobs::GpuKnobs::BLOCK_SIZE>(
        Knobs::GpuKnobs::BLOCK_32 * (2 << gpuBlockSizeExp)
    );
    gpuChunckFactor = static_cast<Knobs::GpuKnobs::CHUNK_FACTOR>(
        Knobs::GpuKnobs::CHUNK_1 * (2 << gpuChunkFactorExp)
    );
    gpuEdgesTextureMem = static_cast<Knobs::GpuKnobs::MEMORY_TYPE>(useGpuEdgesTextureMem);
    gpuOffsetsTextureMem = static_cast<Knobs::GpuKnobs::MEMORY_TYPE>(useGpuOffsetsTextureMem);
}