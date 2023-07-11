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

#include <margot/margot.hpp>

namespace po = boost::program_options;

po::options_description SetupOptions();
void CastKnobs(
    unsigned int deviceId,
    unsigned int gpuBlockSizeExp,
    unsigned int gpuChunkFactorExp,
    unsigned int gpuOffsetsMemId,
    unsigned int gpuEdgesMemId,
    Knobs::DEVICE& device,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize,
    Knobs::GpuKnobs::CHUNK_FACTOR& gpuChunckFactor,
    Knobs::GpuKnobs::MEMORY_TYPE& edgesOffsetsMemType,
    Knobs::GpuKnobs::MEMORY_TYPE& edgesMemType
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
    margot::bfs::context().manager.wait_for_knowledge(10);

    unsigned int deviceId = 0;
    unsigned int gpuBlockSizeExp = 0;
    unsigned int gpuChunkFactorExp = 0;
    unsigned int gpuOffsetsMemId = 0;
    unsigned int gpuEdgesMemId = 0;
    
    Knobs::DEVICE device;
    Knobs::GpuKnobs::BLOCK_SIZE gpuBlockSize;
    Knobs::GpuKnobs::CHUNK_FACTOR gpuChunckFactor;
    Knobs::GpuKnobs::MEMORY_TYPE edgesOffsetsMemType;
    Knobs::GpuKnobs::MEMORY_TYPE edgesMemType;

    unsigned int cpuThreads = 1;
    unsigned int precision = 1;

    CastKnobs(
        deviceId,
        gpuBlockSizeExp,
        gpuChunkFactorExp,
        gpuOffsetsMemId,
        gpuEdgesMemId,
        device,
        gpuBlockSize,
        gpuChunckFactor,
        edgesOffsetsMemType,
        edgesMemType
    );
    
    while(margot::bfs::context().manager.in_design_space_exploration()){

        if(margot::bfs::update(
            cpuThreads,
            deviceId,
            gpuBlockSizeExp,
            gpuChunkFactorExp,
            gpuEdgesMemId,
            gpuOffsetsMemId
        )){
            CastKnobs(
                deviceId,
                gpuBlockSizeExp,
                gpuChunkFactorExp,
                gpuOffsetsMemId,
                gpuEdgesMemId,
                device,
                gpuBlockSize,
                gpuChunckFactor,
                edgesOffsetsMemType,
                edgesMemType
            );
            margot::bfs::context().manager.configuration_applied();
        }

        margot::bfs::start_monitors();

        std::string inputFileURL(vm["input-file"].as<std::string>());
        Graph::Graph graph(Graph::ReadGraphFile(inputFileURL));

        std::unique_ptr<Bfs::Bfs> bfs( 
            (device == Knobs::DEVICE::GPU) ?
            static_cast<Bfs::Bfs*>(new Bfs::BfsCuda(
                graph,
                0,
                static_cast<unsigned int>(gpuBlockSize),
                static_cast<unsigned int>(gpuChunckFactor),
                static_cast<bool>(edgesOffsetsMemType),
                static_cast<bool>(edgesMemType)
            )) :
            static_cast<Bfs::Bfs*>(new Bfs::BfsCpu(
                graph,
                0,
                cpuThreads
            ))
        );

        while(!bfs->run()){}

        std::vector<int> costs = bfs->getResult();

       if(vm.count("output-file")){
            Bfs::WriteCosts(vm["output-file"].as<std::string>(), costs);
        }

        margot::bfs::stop_monitors();
        margot::bfs::push_custom_monitor_values();
        margot::bfs::log();
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

void CastKnobs(
    unsigned int deviceId,
    unsigned int gpuBlockSizeExp,
    unsigned int gpuChunkFactorExp,
    unsigned int edgesOffsetsMemId,
    unsigned int edgesMemId,
    Knobs::DEVICE& device,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize,
    Knobs::GpuKnobs::CHUNK_FACTOR& gpuChunckFactor,
    Knobs::GpuKnobs::MEMORY_TYPE& gpuOffsetsMemId,
    Knobs::GpuKnobs::MEMORY_TYPE& gpuEdgesMemId
){
    device = static_cast<Knobs::DEVICE>(deviceId);
    gpuBlockSize = static_cast<Knobs::GpuKnobs::BLOCK_SIZE>(
        Knobs::GpuKnobs::BLOCK_32 << gpuBlockSizeExp
    );
    gpuChunckFactor = static_cast<Knobs::GpuKnobs::CHUNK_FACTOR>(
        Knobs::GpuKnobs::CHUNK_1 << gpuChunkFactorExp
    );
    gpuOffsetsMemId = static_cast<Knobs::GpuKnobs::MEMORY_TYPE>(
        static_cast<bool>(edgesOffsetsMemId));
    gpuEdgesMemId = static_cast<Knobs::GpuKnobs::MEMORY_TYPE>(
        static_cast<bool>(edgesMemId));
}