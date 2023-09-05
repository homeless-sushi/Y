#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Nbody/Body.h"
#include "Nbody/ReadWrite.h"
#include "Nbody/NbodyCpu.h"
#include "Nbody/NbodyCuda.h"

#include "Knobs/Device.h"
#include "Knobs/Precision.h"

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

    std::vector<Nbody::Body> bodies;
    float targetSimulationTime;
    float targetTimeStep;
    Nbody::ReadBodyFile(inputFileURL, bodies, targetSimulationTime, targetTimeStep);

    unsigned int precision = vm["precision"].as<unsigned int>();
    float actualTimeStep = targetTimeStep;
    float approximateSimulationTime = Knobs::GetApproximateSimTime(
        targetSimulationTime, 
        targetTimeStep,
        precision
    );

    Knobs::DEVICE device = vm["gpu"].as<bool>() ? Knobs::DEVICE::GPU : Knobs::DEVICE::CPU;
    unsigned int cpuThreads = vm["cpu-threads"].as<unsigned int>();
    unsigned int gpuBlockSize = 32 * (2 << vm["gpu-block-exp"].as<unsigned int>());

    std::unique_ptr<Nbody::Nbody> nbody( 
        device == Knobs::DEVICE::GPU ?
        static_cast<Nbody::Nbody*>(new NbodyCuda::NbodyCuda(bodies, targetSimulationTime, actualTimeStep, gpuBlockSize)) :
        static_cast<Nbody::Nbody*>(new NbodyCpu::NbodyCpu(bodies, targetSimulationTime, actualTimeStep, cpuThreads))
    );
    nbody->run();
    bodies = nbody->getResult();

    if(vm.count("output-file")){
        Nbody::WriteCSVFile(vm["output-file"].as<std::string>(), 
            bodies,
            targetSimulationTime,
            targetTimeStep,
            nbody->getSimulatedTime(),
            actualTimeStep,
            precision
        );
    }
    
    return 0;
}

po::options_description SetupOptions()
{
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Display help message")

    ("input-file,I", po::value<std::string>(), "input file body file")
    ("output-file,O", po::value<std::string>(), "output file result file")

    ("gpu", po::bool_switch(), "use gpu")
    ("cpu-threads", po::value<unsigned int>()->default_value(1), "number of cpu threads")
    ("gpu-block-exp", po::value<unsigned int>()->default_value(0), "block exp; block size = 32*2^X")

    ("precision,P", po::value<unsigned int>()->default_value(100), "precision in range 0-100")
    ;

    return desc;
}
