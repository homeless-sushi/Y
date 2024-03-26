#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <csignal>

#include "Histo/Histo.h"
#include "Histo/HistoCuda.h"
#include "Histo/HistoCpu.h"
#include "Histo/ReadWrite.h"

#include "Knobs/Device.h"

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
    unsigned int gpuNBlocksExp,
    unsigned int gpuBlockSizeExp,
    Knobs::GpuKnobs::N_BLOCKS& gpuNBlocks,
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
    unsigned int inputSize = extractSize(vm["input-file"].as<std::string>());
    struct app_data* data = registerAttach(
        vm["instance-name"].as<std::string>().c_str(),
        "HISTO", inputSize,
        targetThroughput,
        4,
        true);
    int dataSemId = semget(getpid(), 1, 0);

    Frequency::CPU_FRQ cpuFreq = Frequency::getMinCpuFreq();
    Frequency::GPU_FRQ gpuFreq = Frequency::getMinGpuFreq();
    unsigned int useGpu = 0;
    unsigned int nCores = 1;
    unsigned int gpuNBlocksExp = 0;
    Knobs::GpuKnobs::N_BLOCKS gpuNBlocks;
    unsigned int gpuBlockSizeExp = 0;
    Knobs::GpuKnobs::BLOCK_SIZE gpuBlockSize;

    CastKnobs(
        gpuNBlocksExp,
        gpuBlockSizeExp,
        gpuNBlocks,
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
        if(margot::histo::update(useGpu, cpuFreq, gpuFreq, inputSize, nCores, gpuBlockSizeExp, gpuNBlocksExp)){   
            CastKnobs(
                gpuNBlocksExp,
                gpuBlockSizeExp,
                gpuNBlocks,
                gpuBlockSize
            );
            margot::histo::context().manager.configuration_applied(); 
        }
        margot::histo::start_monitors();
        //STOP: MARGOT PULL

        //START: WIND UP
        std::string inputFileURL(vm["input-file"].as<std::string>());
        unsigned imgWidth, imgHeight;
        std::vector<unsigned short> rgb;
        Histo::ReadBinaryDataFile(
            inputFileURL,
            imgWidth, imgHeight,
            rgb
        );

        std::unique_ptr<Histo::Histo> histo(
            useGpu == Knobs::DEVICE::GPU ?
            static_cast<Histo::Histo*>(new Histo::HistoCuda(
                rgb,
                gpuNBlocks, gpuBlockSize
            )) :
            static_cast<Histo::Histo*>(new Histo::HistoCpu(
                rgb,
                nCores
            ))
        );
        //STOP: WIND UP

        //START: KERNEL
        histo->run();
        //STOP: KERNEL

        //START: WIND DOWN
        std::vector<unsigned> res = histo->getResult();
        if(vm.count("output-file")){
            Histo::WriteHistogramFile(
                vm["output-file"].as<std::string>(),
                res
            );
        }
        //START: WIND DOWN

        //START: MARGOT PUSH
        margot::histo::stop_monitors();
        margot::histo::log();
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
    ("input-file,I", po::value<std::string>(), "input file body file")
    ("output-file,O", po::value<std::string>(), "output file result file")
    ("instance-name", po::value<std::string>()->default_value("HISTO"), "name of benchmark instance")
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
    unsigned int gpuNBlocksExp,
    unsigned int gpuBlockSizeExp,
    Knobs::GpuKnobs::N_BLOCKS& gpuNBlocks,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize
)
{
    gpuNBlocks = Knobs::nBlocksfromExponent(gpuNBlocksExp);
    gpuBlockSize = Knobs::blockSizefromExponent(gpuBlockSizeExp);
}
