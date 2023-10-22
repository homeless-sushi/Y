#include <iostream>
#include <string>
#include <memory>

#include "Graph/Graph.h"
#include "Graph/ReadWrite.h"

#include "Bfs/Bfs.h"
#include "Bfs/BfsCpu.h"
#include "Bfs/BfsCuda.h"
#include "Bfs/ReadWrite.h"

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
    Graph::Graph graph(Graph::ReadGraphFile(inputFileURL));

    Knobs::DEVICE device = vm["gpu"].as<bool>() ? Knobs::DEVICE::GPU : Knobs::DEVICE::CPU;
    unsigned int cpuThreads = vm["cpu-threads"].as<unsigned int>();
    unsigned int gpuBlockSize = 32 * (2 << vm["gpu-block-exp"].as<unsigned int>());
    unsigned int gpuChunkSize = 1 * (2 << vm["gpu-chunk-exp"].as<unsigned int>());
    bool edgesTextureMem = vm["gpu-edges-texture-mem"].as<bool>();
    bool offsetsTextureMem = vm["gpu-offsets-texture-mem"].as<bool>();
    std::unique_ptr<Bfs::Bfs> bfs( 
        device == Knobs::DEVICE::GPU ?
        static_cast<Bfs::Bfs*>(new Bfs::BfsCuda(
            graph,
            0,
            static_cast<unsigned int>(gpuBlockSize),
            static_cast<unsigned int>(gpuChunkSize),
            static_cast<bool>(edgesTextureMem),
            static_cast<bool>(offsetsTextureMem)
        )) :
        static_cast<Bfs::Bfs*>(new Bfs::BfsCpu(
            graph, 
            0, 
            cpuThreads
        ))
    );

    bfs->run();

    std::vector<int> costs = bfs->getResult();

    if(vm.count("output-file")){
        Bfs::WriteCosts(vm["output-file"].as<std::string>(), costs);
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
    ("gpu-block-exp", po::value<unsigned int>()->default_value(0), "block exp; block size = 32*2^X")
    ("gpu-chunk-exp", po::value<unsigned int>()->default_value(0), "chunk exp; chunk size = 1*2^X")
    ("gpu-edges-texture-mem", po::bool_switch(), "use gpu texture memory for edges")
    ("gpu-offsets-texture-mem", po::bool_switch(), "use gpu texture memory for offsets")
    ;

    return desc;
}
