#include <iostream>
#include <string>

#include <Atom/Atom.h>
#include <Atom/ReadWrite.h>
#include <Atom/Utils.h>

#include <Cutcp/CutcpCpu.h>
#include <Cutcp/Lattice.h>
#include <Cutcp/ReadWrite.h>

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
    std::cout << "extent of domain is:" << std::endl;
    std::cout << "\t" << "minimum: " << minCoords.x << " " << minCoords.y << " " << minCoords.z << std::endl;
    std::cout << "\t" << "maximum: " << maxCoords.x << " " << maxCoords.y << " " << maxCoords.z << std::endl;

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
    CutcpCpu::CutcpCpu cutcp(
        lattice,
        atoms,
        cutoff,
        exclusionCutoff,
        16
    );
    cutcp.run();
    lattice = cutcp.getResult();
    
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
    ("precision,P", po::value<unsigned int>()->default_value(100), "precision in range 0-100")
    ;

    return desc;
}
