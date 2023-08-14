#include <iostream>
#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include <Atom/Atom.h>
#include <Atom/ReadWrite.h>
#include <Atom/Utils.h>

#include <Cutcp/CutcpCpu.h>
#include <Cutcp/CutcpCuda.h>
#include <Cutcp/Lattice.h>
#include <Cutcp/ReadWrite.h>

#include <Knobs/Device.h>
#include <Knobs/Precision.h>

#include <Dvfs/Dvfs.h>

#include <boost/program_options.hpp>

#include <margot/margot.hpp>

namespace po = boost::program_options;

po::options_description SetupOptions();
void CastKnobs(
    unsigned int deviceId,
    unsigned int gpuBlockSizeExp,
    unsigned int inputSize,
    Knobs::DEVICE& device,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize,
    std::string& inputUrl
);

int main(int argc, char *argv[])
{
    po::options_description desc(SetupOptions());
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 0;
    }

    margot::init();
    margot::cutcp::context().manager.wait_for_knowledge(10);

    unsigned int deviceId = 0;
    unsigned int gpuBlockSizeExp = 0;
    unsigned int inputSize = 512;
    
    Knobs::DEVICE device;
    Knobs::GpuKnobs::BLOCK_SIZE gpuBlockSize;
    std::string inputUrl;

    unsigned int cpuFreq = Dvfs::CPU_FRQ::FRQ_102000KHz;
    unsigned int gpuFreq = Dvfs::GPU_FRQ::FRQ_76800000Hz;
    unsigned int cpuThreads = 1;
    unsigned int precision = 1;

    CastKnobs(
        deviceId,
        gpuBlockSizeExp,
        inputSize,
        device,
        gpuBlockSize,
        inputUrl
    );

    while(margot::cutcp::context().manager.in_design_space_exploration()){

        if(margot::cutcp::update(
            cpuFreq,
            cpuThreads,
            deviceId,
            gpuBlockSizeExp,
            gpuFreq,
            inputSize,
            precision))
        {
            CastKnobs(
                deviceId,
                gpuBlockSizeExp,
                inputSize,
                device,
                gpuBlockSize,
                inputUrl
            );
            margot::cutcp::context().manager.configuration_applied();
        }
        
        Dvfs::SetCpuFreq(static_cast<Dvfs::CPU_FRQ>(cpuFreq));
        Dvfs::SetGpuFreq(static_cast<Dvfs::GPU_FRQ>(gpuFreq));
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        margot::cutcp::start_monitors();

        std::vector<Atom::Atom> atoms = Atom::ReadAtomFile(inputUrl);
    
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
        cutcp->run();
        lattice = cutcp->getResult();

        if(vm.count("output-file")){
            Cutcp::WriteLattice(vm["output-file"].as<std::string>(), lattice);
        }

        margot::cutcp::stop_monitors();
        margot::cutcp::push_custom_monitor_values();
        margot::cutcp::log();
    }

    return 0;
}

po::options_description SetupOptions()
{
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Display help message")
    ("output-file,O", po::value<std::string>(), "output lattice result file")
    ;

    return desc;
}

void CastKnobs(
    unsigned int deviceId,
    unsigned int gpuBlockSizeExp,
    unsigned int inputSize,
    Knobs::DEVICE& device,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize,
    std::string& inputUrl
)
{
    device = static_cast<Knobs::DEVICE>(deviceId);
    gpuBlockSize = static_cast<Knobs::GpuKnobs::BLOCK_SIZE>(
        Knobs::GpuKnobs::BLOCK_32 << gpuBlockSizeExp
    );
    std::stringstream inputUrlStream;
    inputUrlStream << "/home/miele/Vivian/Thesis/apps/Y/programs/CUTCP/data/in/" << inputSize << ".txt";
    inputUrl = inputUrlStream.str();
}