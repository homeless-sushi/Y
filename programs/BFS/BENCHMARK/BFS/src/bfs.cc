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

#include <margot/margot.hpp>

namespace po = boost::program_options;

po::options_description SetupOptions();
void SetupSignals();
void CastKnobs(
    unsigned int deviceId,
    unsigned int gpuBlockSizeExp,
    unsigned int gpuChunkFactorExp,
    unsigned int gpuOffsetsMemId,
    unsigned int gpuEdgesMemId,
    Knobs::DEVICE& device,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize,
    Knobs::GpuKnobs::CHUNK_FACTOR& gpuChunckFactor,
    Knobs::GpuKnobs::MEMORY_TYPE& edgesOffsetsMemType,
    Knobs::GpuKnobs::MEMORY_TYPE& edgesMemType
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
    unsigned int gpuBlockSizeExp = 0;
    unsigned int gpuChunkFactorExp = 0;
    unsigned int gpuOffsetsMemId = 0;
    unsigned int gpuEdgesMemId = 0;
    
    Knobs::DEVICE device;
    Knobs::GpuKnobs::BLOCK_SIZE gpuBlockSize;
    Knobs::GpuKnobs::CHUNK_FACTOR gpuChunckFactor;
    Knobs::GpuKnobs::MEMORY_TYPE edgesOffsetsMemType;
    Knobs::GpuKnobs::MEMORY_TYPE edgesMemType;

    unsigned int cpuThreads = 1;
    unsigned int precision = 1;

    CastKnobs(
        deviceId,
        gpuBlockSizeExp,
        gpuChunkFactorExp,
        gpuOffsetsMemId,
        gpuEdgesMemId,
        device,
        gpuBlockSize,
        gpuChunckFactor,
        edgesOffsetsMemType,
        edgesMemType
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
        if(margot::bfs::update(
            cpuThreads,
            deviceId,
            gpuBlockSizeExp,
            gpuChunkFactorExp,
            gpuEdgesMemId,
            gpuOffsetsMemId
        )){
            CastKnobs(
                deviceId,
                gpuBlockSizeExp,
                gpuChunkFactorExp,
                gpuOffsetsMemId,
                gpuEdgesMemId,
                device,
                gpuBlockSize,
                gpuChunckFactor,
                edgesOffsetsMemType,
                edgesMemType
            );
            margot::bfs::context().manager.configuration_applied();
        }
        std::cout << "MARGOT PULL,STOP,CPU," << now() << std::endl;
        //STOP: MARGOT PULL

        margot::bfs::start_monitors();

        //START: WIND UP
        std::cout << "WIND UP,START,CPU," << now() << std::endl;
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
        std::cout << "WIND UP,STOP,CPU," << now() << std::endl;
        //STOP: WIND UP

        //START: KERNEL
        std::cout << "KERNEL,START," << Knobs::DeviceToString(device) << "," << now() << std::endl;
        while(!bfs->run()){}
        std::vector<int> costs = bfs->getResult();
        std::cout << "KERNEL,STOP," << Knobs::DeviceToString(device) << "," << now() << std::endl;
        //STOP: KERNEL

        while(!bfs->run()){}

        std::vector<int> costs = bfs->getResult();

        //START: WIND DOWN
        std::cout << "WIND DOWN,START,CPU," << now() << std::endl;
        if(vm.count("output-file")){
            Bfs::WriteCosts(vm["output-file"].as<std::string>(), costs);
        }
        std::cout << "WIND DOWN,STOP,CPU," << now() << std::endl;
        //START: WIND DOWN



        //START: MARGOT PUSH
        std::cout << "MARGOT PUSH,START,CPU," << now() << std::endl;
        margot::bfs::stop_monitors();
        margot::bfs::push_custom_monitor_values();
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
    ("input-file,I", po::value<std::string>(), "input atoms file")
    ("output-file,O", po::value<std::string>(), "output lattice result file")
    ("instance-name", po::value<std::string>()->default_value("CUTCP"), "name of benchmark instance")
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
    unsigned int deviceId,
    unsigned int gpuBlockSizeExp,
    unsigned int gpuChunkFactorExp,
    unsigned int edgesOffsetsMemId,
    unsigned int edgesMemId,
    Knobs::DEVICE& device,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize,
    Knobs::GpuKnobs::CHUNK_FACTOR& gpuChunckFactor,
    Knobs::GpuKnobs::MEMORY_TYPE& gpuOffsetsMemId,
    Knobs::GpuKnobs::MEMORY_TYPE& gpuEdgesMemId
){
    device = static_cast<Knobs::DEVICE>(deviceId);
    gpuBlockSize = static_cast<Knobs::GpuKnobs::BLOCK_SIZE>(
        Knobs::GpuKnobs::BLOCK_32 << gpuBlockSizeExp
    );
    gpuChunckFactor = static_cast<Knobs::GpuKnobs::CHUNK_FACTOR>(
        Knobs::GpuKnobs::CHUNK_1 << gpuChunkFactorExp
    );
    gpuOffsetsMemId = static_cast<Knobs::GpuKnobs::MEMORY_TYPE>(
        static_cast<bool>(edgesOffsetsMemId));
    gpuEdgesMemId = static_cast<Knobs::GpuKnobs::MEMORY_TYPE>(
        static_cast<bool>(edgesMemId));
}