#include <iostream>
#include <chrono>
#include <memory>
#include <string>

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

namespace po = boost::program_options;

po::options_description SetupOptions();

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

    const auto appStart = std::chrono::system_clock::now();
    const auto inputStart = appStart;

    std::string inputFileURL(vm["input-file"].as<std::string>());
    std::vector<Atom::Atom> atoms = Atom::ReadAtomFile(inputFileURL);

    const auto inputStop = std::chrono::system_clock::now();
    
    Vector::Vec3 minCoords;
    Vector::Vec3 maxCoords;
    Atom::GetAtomBounds(atoms, minCoords, maxCoords);

    float padding = 0.5;
    Vector::Vec3 paddingVec(padding);
    minCoords = minCoords - paddingVec;
    maxCoords = maxCoords + paddingVec;

    float spacing = 0.5;
    Lattice::Lattice lattice(minCoords, maxCoords, spacing);

    float cutoff = Knobs::GetCutoff(minCoords, maxCoords, spacing, vm["precision"].as<unsigned int>());
    float exclusionCutoff = 1.;

    Knobs::DEVICE device = vm["gpu"].as<bool>() ? Knobs::DEVICE::GPU : Knobs::DEVICE::CPU;
    unsigned int cpuThreads = vm["cpu-threads"].as<unsigned int>();
    unsigned int gpuBlockSize = 32 * (2 << vm["gpu-block-exp"].as<unsigned int>());

    std::unique_ptr<Cutcp::Cutcp> cutcp( 
            device == Knobs::DEVICE::GPU ?
            static_cast<Cutcp::Cutcp*>(new CutcpCuda::CutcpCuda(lattice, atoms, cutoff, exclusionCutoff, gpuBlockSize)) :
            static_cast<Cutcp::Cutcp*>(new CutcpCpu::CutcpCpu(lattice, atoms, cutoff, exclusionCutoff, cpuThreads))
        );
    cutcp->run();
    lattice = cutcp->getResult();

    const auto outputStart = std::chrono::system_clock::now();

    if(vm.count("output-file")){
        Cutcp::WriteLattice(vm["output-file"].as<std::string>(), lattice);
    }

    const auto outputStop = std::chrono::system_clock::now();
    auto appStop = std::chrono::system_clock::now();


    auto appTotalTime = std::chrono::duration<double, std::milli>((appStop - appStart)).count();
    auto inputTotalTime = std::chrono::duration<double, std::milli>((inputStop - inputStart)).count();
    auto outputTotalTime = std::chrono::duration<double, std::milli>((outputStop - outputStart)).count();

    std::cout << "APP_TIME " << appTotalTime << "\n"
        << "INPUT_TIME " << inputTotalTime << "\n"
        << "OUTPUT_TIME " << outputTotalTime << "\n";
    
    if(device == Knobs::DEVICE::GPU){

        Cutcp::Cutcp* cutcpPointer(cutcp.get());
        cutcp.release();
        std::unique_ptr<CutcpCuda::CutcpCuda> cutcpCuda( 
            dynamic_cast<CutcpCuda::CutcpCuda*>(cutcpPointer)
        );

        float dataUploadTime = cutcpCuda->getDataUploadTime();
        float kernelTime = cutcpCuda->getKernelTime();
        float dataDownloadTime = cutcpCuda->getDataDownloadTime();

        std::cout << "GPU_UPLOAD_TIME " << dataUploadTime << "\n"
            << "GPU_KERNEL_TIME " << kernelTime << "\n"
            << "GPU_DOWNLOAD_TIME " << dataDownloadTime << "\n";

    }

    std::cout << std::endl;

    return 0;
}

po::options_description SetupOptions()
{
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Display help message")

    ("input-file,I", po::value<std::string>(), "input atoms file")
    ("output-file,O", po::value<std::string>(), "output lattice result file")

    ("gpu", po::bool_switch(), "use gpu")
    ("cpu-threads", po::value<unsigned int>()->default_value(1), "number of cpu threads")
    ("gpu-block-exp", po::value<unsigned int>()->default_value(0), "block exp; block size = 32*2^X")

    ("precision,P", po::value<unsigned int>()->default_value(100), "precision in range 0-100")
    ;

    return desc;
}
