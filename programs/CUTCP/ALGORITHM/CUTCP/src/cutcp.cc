#include <iostream>
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

    std::string inputFileURL(vm["input-file"].as<std::string>());
    std::vector<Atom::Atom> atoms = Atom::ReadAtomFile(inputFileURL);
    
    Vector::Vec3 minCoords;
    Vector::Vec3 maxCoords;
    Atom::GetAtomBounds(atoms, minCoords, maxCoords);

    float padding = 0.5;
    Vector::Vec3 paddingVec(padding);
    minCoords = minCoords - paddingVec;
    maxCoords = maxCoords + paddingVec;
    std::cout << "padding domain by " << padding << " Angstroms:" << std::endl;
    std::cout << "domain lenghts are " << maxCoords.x-minCoords.x << " by " << maxCoords.y-minCoords.y << " by " << maxCoords.z-minCoords.z << std::endl;

    float spacing = 0.5;
    Lattice::Lattice lattice(minCoords, maxCoords, spacing);

    float cutoff = Knobs::GetCutoff(minCoords, maxCoords, spacing, vm["precision"].as<unsigned int>());
    float exclusionCutoff = 1.;
    std::cout << "potential cutoff is " << cutoff << std::endl;
    std::cout << "exclusion cutoff is " << exclusionCutoff << std::endl;

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

    if(vm.count("output-file")){
        Cutcp::WriteLattice(vm["output-file"].as<std::string>(), lattice);
    }
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
