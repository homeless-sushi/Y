#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <csignal>

#include "Sha/Sha.h"

#include <boost/program_options.hpp>

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

    long double targetThroughput = vm["target-throughput"].as<long double>();
    //Attach to controller
    struct app_data* data = registerAttach(
        vm["instance-name"].as<std::string>().c_str(),
        targetThroughput,
        1,
        false);
    int dataSemId = semget(getpid(), 1, 0);
    //STOP: SETUP

    //START: WAIT REGISTRATION
    //Spinlock
    while(true){
        if(isRegistered(data)){
            setTickStartTime(data);
            break;
        }
    }
    //STOP: WAIT REGISTRATION

    bool error = false;
    while(!stop && !error){

        Sha::ShaInfo sha(0);
        sha.digestFile(vm["input-file"].as<std::string>());

        if(vm.count("output-file")){
            std::string outFileUrl(vm["output-file"].as<std::string>());
            std::ofstream outFile(outFileUrl);
            if (!outFile.is_open()){
                throw std::runtime_error("Cannot open file: " + outFileUrl);
            }
            outFile << sha << std::endl;
        }

        //START: CONTROLLER PUSH
        //Add tick
        autosleep(data, targetThroughput);
        error = binarySemaphoreWait(dataSemId);
        addTick(data, 1);
        error = binarySemaphorePost(dataSemId);
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
        std::cerr << "\n";
        std::cerr << "Received signal: " << signal << "\n";
        std::cerr << "Stopping benchmark" << "\n";
        std::cerr << std::endl;

        stop = true;
    };

    std::signal(SIGINT, stopBenchmark);
    std::signal(SIGTERM, stopBenchmark);
}

