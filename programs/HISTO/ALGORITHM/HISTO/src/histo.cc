#include <iostream>
#include <string>
#include <memory>

#include "Histo/Histo.h"
#include "Histo/HistoCuda.h"
#include "Histo/HistoCpu.h"
#include "Histo/ReadWrite.h"

#include "Knobs/Device.h"

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
    unsigned imgWidth, imgHeight;
    unsigned histoWidth, histoHeight;
    std::vector<unsigned short> rgb;
    Histo::ReadBinaryDataFile(
        inputFileURL,
        imgWidth, imgHeight,
        rgb
    );

    Knobs::DEVICE device = vm["gpu"].as<bool>() ? Knobs::DEVICE::GPU : Knobs::DEVICE::CPU;
    unsigned int cpuThreads = vm["cpu-threads"].as<unsigned int>();
    unsigned int gpuNBlocks = static_cast<unsigned int>(Knobs::nBlocksfromExponent(vm["gpu-n-blocks-exp"].as<unsigned int>()));
    unsigned int gpuBlockSize = static_cast<unsigned int>(Knobs::blockSizefromExponent(vm["gpu-block-size-exp"].as<unsigned int>()));
    std::unique_ptr<Histo::Histo> histo(
        device == Knobs::DEVICE::GPU ?
        static_cast<Histo::Histo*>(new Histo::HistoCuda(
            rgb,
            gpuNBlocks, gpuBlockSize
        )) :
        static_cast<Histo::Histo*>(new Histo::HistoCpu(
            rgb,
            cpuThreads
        ))
    );
    
    unsigned times = vm["times"].as<unsigned>();
    for(unsigned i = 0; i < times; i++){
        histo->run();
    }

    std::vector<unsigned> res = histo->getResult();

    if(vm.count("output-file")){
        Histo::WriteHistogramFile(
            vm["output-file"].as<std::string>(),
            res
        );
    }

    return 0;
}

po::options_description SetupOptions()
{
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Display help message")

    ("input-file,I", po::value<std::string>(), "input file with graph description")
    ("output-file,O", po::value<std::string>(), "output file with bfs solution")

    ("gpu", po::bool_switch(), "use gpu")
    ("cpu-threads", po::value<unsigned int>()->default_value(1), "number of cpu threads")
    ("gpu-n-blocks-exp", po::value<unsigned int>()->default_value(0), "n blocks exp; n blocks = 8*2^X")
    ("gpu-block-size-exp", po::value<unsigned int>()->default_value(0), "block exp; block size = 32*2^X")

    ("times,T", po::value<unsigned int>()->default_value(1), "repeat the kernel T times")
    ;

    return desc;
}
