#include <iostream>
#include <string>
#include <memory>

#include "Graph/Graph.h"
#include "Graph/ReadWrite.h"

#include "Bfs/Bfs.h"
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

    Bfs::BfsCuda bfs(
        graph,
        0,
        static_cast<unsigned int>(Knobs::GpuKnobs::BLOCK_SIZE::BLOCK_32),
        static_cast<unsigned int>(Knobs::GpuKnobs::CHUNK_FACTOR::CHUNK_1),
        static_cast<bool>(Knobs::GpuKnobs::MEMORY_TYPE::DEVICE_MEM),
        static_cast<bool>(Knobs::GpuKnobs::MEMORY_TYPE::TEXTURE_MEM)
    );
    
    while(!bfs.run()){}

    std::vector<int> costs = bfs.getResult();

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
    ;

    return desc;
}
