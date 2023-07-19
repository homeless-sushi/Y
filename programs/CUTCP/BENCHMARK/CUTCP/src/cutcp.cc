#include <iostream>
#include <memory>
#include <string>

#include <csignal>

#include <Atom/Atom.h>
#include <Atom/ReadWrite.h>
#include <Atom/Utils.h>

#include <Cutcp/CutcpCpu.h>
#include <Cutcp/CutcpCuda.h>
#include <Cutcp/Lattice.h>
#include <Cutcp/ReadWrite.h>

#include <Knobs/Device.h>
#include <Knobs/Precision.h>

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
    unsigned int gpuBlockSizeExp,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize
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
    
    Knobs::DEVICE device;
    Knobs::GpuKnobs::BLOCK_SIZE gpuBlockSize;

    unsigned int cpuThreads = 1;
    unsigned int precision = 1;

    CastKnobs(
        gpuBlockSizeExp,
        gpuBlockSize
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
        if(margot::cutcp::update(cpuThreads, deviceId, gpuBlockSizeExp, precision)){
            CastKnobs(
                gpuBlockSizeExp,
                gpuBlockSize
            );
            margot::cutcp::context().manager.configuration_applied();
        }
        margot::cutcp::start_monitors();
        std::cout << "MARGOT PULL,STOP,CPU," << now() << std::endl;
        //STOP: MARGOT PULL

        //START: WIND UP
        std::cout << "WIND UP,START,CPU," << now() << std::endl;
        std::string inputFileURL(vm["input-file"].as<std::string>());
        std::vector<Atom::Atom> atoms = Atom::ReadAtomFile(inputFileURL);
    
        Vector::Vec3 minCoords;
        Vector::Vec3 maxCoords;
        Atom::GetAtomBounds(atoms, minCoords, maxCoords);
        float padding = 0.5;
        Vector::Vec3 paddingVec(padding);
        minCoords = minCoords - paddingVec;
        maxCoords = maxCoords + paddingVec;
        float spacing = 0.5;
        Lattice::Lattice lattice(minCoords, maxCoords, spacing);

        float cutoff = Knobs::GetCutoff(minCoords, maxCoords, spacing, precision);
        float exclusionCutoff = 1.;

        std::unique_ptr<Cutcp::Cutcp> cutcp( 
                device == Knobs::DEVICE::GPU ?
                static_cast<Cutcp::Cutcp*>(new CutcpCuda::CutcpCuda(lattice, atoms, cutoff, exclusionCutoff, gpuBlockSize)) :
                static_cast<Cutcp::Cutcp*>(new CutcpCpu::CutcpCpu(lattice, atoms, cutoff, exclusionCutoff, cpuThreads))
            );
        std::cout << "WIND UP,STOP,CPU," << now() << std::endl;
        //STOP: WIND UP

        //START: KERNEL
        std::cout << "KERNEL,START," << Knobs::DeviceToString(device) << "," << now() << std::endl;
        cutcp->run();
        std::cout << "KERNEL,STOP," << Knobs::DeviceToString(device) << "," << now() << std::endl;
        //STOP: KERNEL

        //START: WIND DOWN
        std::cout << "WIND DOWN,START,CPU," << now() << std::endl;
        lattice = cutcp->getResult();
        if(vm.count("output-file")){
            Cutcp::WriteLattice(vm["output-file"].as<std::string>(), lattice);
        }
        std::cout << "WIND DOWN,STOP,CPU," << now() << std::endl;
        //START: WIND DOWN

        //START: MARGOT PUSH
        std::cout << "MARGOT PUSH,START,CPU," << now() << std::endl;
        margot::cutcp::stop_monitors();
        margot::cutcp::push_custom_monitor_values();
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
    unsigned int gpuBlockSizeExp,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize
)
{
    gpuBlockSize = static_cast<Knobs::GpuKnobs::BLOCK_SIZE>(
        Knobs::GpuKnobs::BLOCK_32 << gpuBlockSizeExp
    );
}