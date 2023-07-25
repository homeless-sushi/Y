#include <iostream>
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
    std::cout << "EVENT,TYPE,DEVICE,TIMESTAMP" << "\n";
 
    //START: SETUP
    std::cout << "SETUP,START,CPU," << now() << "\n";
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

    std::cout << "SETUP,STOP,CPU," << now() << "\n";
    //STOP: SETUP

    //Spinlock
    //START: WAIT REGISTRATION
    std::cout << "WAIT REGISTRATION,START,CPU," << now() << "\n";
    while(true){
        if(isRegistered(data)){
            setTickStartTime(data);
            break;
        }
    }
    std::cout << "WAIT REGISTRATION,STOP,CPU," << now() << "\n";
    //STOP: WAIT REGISTRATION

    bool error = false;
    while(!stop && !error){

        //Read knobs
        //START: CONTROLLER PULL
        std::cout << "CONTROLLER PULL,START,CPU," << now() << "\n";
        error = binarySemaphoreWait(dataSemId);
        error = binarySemaphorePost(dataSemId);
        std::cout << "CONTROLLER PULL,STOP,CPU," << now() << "\n";
        //STOP: CONTROLLER PULL

        //START: WIND UP
        std::cout << "WIND UP,START,CPU," << now() << "\n";
        std::vector<float> fileData;
        std::string inputFileURL(vm["input-file"].as<std::string>());
        Dummy::ReadFile(inputFileURL, fileData);

        cudaDeviceProp deviceProp;
        CudaErrorCheck(cudaGetDeviceProperties(&deviceProp, 0));
        unsigned int smCount = deviceProp.multiProcessorCount;
        unsigned int warpSize = deviceProp.warpSize;
        unsigned int nThreads = smCount*warpSize;
        std::vector<float> dummyData(nThreads);
        std::copy(fileData.begin(), fileData.begin() + nThreads, dummyData.begin());

        Dummy::Dummy dummy(dummyData, smCount, warpSize, vm["times"].as<unsigned int>());
        std::cout << "WIND UP,STOP,CPU," << now() << "\n";
        //STOP: WIND UP

        //START: KERNEL
        std::cout << "KERNEL,START,GPU," << now() << "\n";
        dummy.run();
        std::cout << "KERNEL,STOP,GPU," << now() << "\n";
        //STOP: KERNEL
        
        //START: WIND DOWN
        dummyData = dummy.getResult();
        std::cout << "WIND DOWN,START,CPU," << now() << "\n";
        if(vm.count("output-file")){
            std::string outputFileURL(vm["output-file"].as<std::string>());
            Dummy::WriteFile(outputFileURL, dummyData);
        }
        std::cout << "WIND DOWN,STOP,CPU," << now() << "\n";
        //START: WIND DOWN

        //Add tick
        //START: CONTROLLER PUSH
        std::cout << "CONTROLLER PUSH,START,CPU," << now() << "\n";
        autosleep(data, targetThroughput);
        error = binarySemaphoreWait(dataSemId);
        addTick(data, 1);
        error = binarySemaphorePost(dataSemId);
        std::cout << "CONTROLLER PUSH,STOP,CPU," << now() << "\n";
        //STOP: CONTROLLER PUSH
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
    ("times,T", po::value<unsigned int>()->default_value(1), "knob for CUDA kernel duration")
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
