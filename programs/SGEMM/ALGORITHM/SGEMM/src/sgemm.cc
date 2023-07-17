#include <iostream>
#include <string>
#include <memory>

#include "Sgemm/Matrix.h"
#include "Sgemm/Sgemm.h"
#include "Sgemm/SgemmCpu.h"
#include "Sgemm/SgemmCuda.h"
#include "Sgemm/ReadWrite.h"

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

    std::string inputAUrl(vm["input-A"].as<std::string>());
    std::string inputBUrl(vm["input-B"].as<std::string>());
    std::string inputCUrl(vm["input-C"].as<std::string>());
    Sgemm::Matrix a(Sgemm::ReadMatrixFile(inputAUrl));
    Sgemm::Matrix b(Sgemm::ReadMatrixFile(inputBUrl));
    Sgemm::Matrix c(Sgemm::ReadMatrixFile(inputCUrl));

    //Sgemm::SgemmCpu sgemm(1,1,a,b,c,1,4);
    Sgemm::SgemmCuda sgemm(1,1,a,b,c,16);
    sgemm.run();
    
    Sgemm::Matrix res(sgemm.getResult());
    if(vm.count("output-file")){
        Sgemm::WriteMatrixFile(vm["output-file"].as<std::string>(), res);
    }
    
    return 0;
}

po::options_description SetupOptions()
{
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Display help message")
    ("input-A,A", po::value<std::string>(), "input file with matrix A")
    ("input-B,B", po::value<std::string>(), "input file with matrix B")
    ("input-C,C", po::value<std::string>(), "input file with matrix C")
    ("output-file,O", po::value<std::string>(), "ouput file with result matrix")
    ;

    return desc;
}
