#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "Sha/Sha.h"

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
    return 0;
}

po::options_description SetupOptions()
{
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Display help message")

    ("input-file,I", po::value<std::string>(), "input atoms file")
    ("output-file,O", po::value<std::string>(), "output lattice result file")
    ;

    return desc;
}
