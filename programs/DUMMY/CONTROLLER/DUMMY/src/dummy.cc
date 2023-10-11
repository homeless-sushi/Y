#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <csignal>

#include <cuda_runtime.h>

#include <boost/program_options.hpp>

#include "Dummy/Dummy.h"
#include "Dummy/ReadWrite.h"

#include "CudaError/CudaError.h"

#include "AppRegisterCommon/AppRegister.h"
#include "AppRegisterCommon/Semaphore.h"

#include "AppRegisterClient/AppRegister.h"
#include "AppRegisterClient/Utils.h"

namespace po = boost::program_options;

po::options_description SetupOptions();
void SetupSignals();

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
        true);
    int dataSemId = semget(getpid(), 1, 0);
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
        error = binarySemaphorePost(dataSemId);
        stopTime = std::chrono::system_clock::now();
        duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
        std::cout << "CONTROLLER PULL,CPU," << duration.count() << "\n";
        //STOP: CONTROLLER PULL

        //START: WIND UP
        startTime = std::chrono::system_clock::now();
        std::vector<float> fileData;
        std::string inputFileURL(vm["input-file"].as<std::string>());
        Dummy::ReadFile(inputFileURL, fileData);

        cudaDeviceProp deviceProp;
        CudaErrorCheck(cudaGetDeviceProperties(&deviceProp, 0));
        unsigned int blocks = vm["blocks"].as<unsigned int>();
        unsigned int threads = vm["threads"].as<unsigned int>();
        unsigned int totalThreads = blocks*threads;
        std::vector<float> dummyData(totalThreads);
        std::copy(fileData.begin(), fileData.begin() + totalThreads, dummyData.begin());

        Dummy::Dummy dummy(dummyData, blocks, threads, vm["times"].as<unsigned int>());
        stopTime = std::chrono::system_clock::now();
        duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
        double dataUploadTime = dummy.getDataUploadTime();
        double windUpTime = duration.count() - dataUploadTime;
        std::cout << "WIND UP,CPU," << windUpTime << "\n";
        std::cout << "UPLOAD,GPU," << dataUploadTime<< "\n"; 
        //STOP: WIND UP

        //START: KERNEL
        startTime = std::chrono::system_clock::now();
        if(vm["infinite"].as<bool>()){
            dummy.runInfinite();
        }else{
            dummy.run();
        }
        stopTime = std::chrono::system_clock::now();
        duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
        std::cout << "KERNEL,GPU," << duration.count() << "\n";
        //STOP: KERNEL
        
        //START: WIND DOWN
        startTime = std::chrono::system_clock::now();
        dummyData = dummy.getResult();
        if(vm.count("output-file")){
            std::string outputFileURL(vm["output-file"].as<std::string>());
            Dummy::WriteFile(outputFileURL, dummyData);
        }
        stopTime = std::chrono::system_clock::now();
        duration = std::chrono::duration<double, std::milli>((stopTime - startTime));
        double dataDownloadTime = dummy.getDataDownloadTime();
        double windDownTime = duration.count() - dataDownloadTime;
        std::cout << "DOWNLOAD,GPU," << dataDownloadTime << "\n";
        std::cout << "WIND DOWN,CPU," << windDownTime << "\n";
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
    ("input-file,I", po::value<std::string>(), "input file body file")
    ("output-file,O", po::value<std::string>(), "output file result file")
    ("instance-name", po::value<std::string>()->default_value("DUMMY"), "name of benchmark instance")
    ("target-throughput", po::value<long double>()->default_value(1.0), "target throughput for the kernel")
    ("infinite", po::bool_switch(), "run infinite kernel")
    ("times,T", po::value<unsigned int>()->default_value(1), "knob for CUDA kernel duration")
    ("blocks,N", po::value<unsigned int>()->default_value(1), "number of blocks")
    ("threads,M", po::value<unsigned int>()->default_value(128), "number of threads per block")
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
