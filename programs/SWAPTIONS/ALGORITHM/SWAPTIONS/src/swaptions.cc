#include <iostream>

#include "Swaptions/NumericalRecipes.h"
#include "Swaptions/Probability.h"
#include "Swaptions/ReadWrite.h"
#include "Swaptions/Swaptions.h"

#include "SwaptionsCpu/Kernel.h"

#include "SwaptionsGpu/Kernel.h"

#include "Knobs/Precision.h"

#include <boost/program_options.hpp>

int iN = 11;
int iFactors = 3;
parm *swaptions;

long seed = 1979; // arbitrary (but constant) default value (birth year of Christian Bienia)
long swaption_seed;

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

    unsigned int nSwaptions = vm["swaptions"].as<unsigned int>();
    unsigned int nTrials = vm["trials"].as<unsigned int>();

    unsigned int cpuThreads = vm["cpu-threads"].as<unsigned int>();
    unsigned int cpuTileSize = 16*(2 << vm["cpu-tile-exp"].as<unsigned int>());
    unsigned int gpuBlockSize = 32*(2 << vm["gpu-block-exp"].as<unsigned int>());

    FTYPE **factors = NULL;

    swaption_seed = (long)(2147483647L * RanUnif(&seed));

    // INPUT:
    // initialize input dataset
    factors = dmatrix(0, iFactors - 1, 0, iN - 2);
    // the three rows store vol data for the three factors
    factors[0][0] = .01;
    factors[0][1] = .01;
    factors[0][2] = .01;
    factors[0][3] = .01;
    factors[0][4] = .01;
    factors[0][5] = .01;
    factors[0][6] = .01;
    factors[0][7] = .01;
    factors[0][8] = .01;
    factors[0][9] = .01;

    factors[1][0] = .009048;
    factors[1][1] = .008187;
    factors[1][2] = .007408;
    factors[1][3] = .006703;
    factors[1][4] = .006065;
    factors[1][5] = .005488;
    factors[1][6] = .004966;
    factors[1][7] = .004493;
    factors[1][8] = .004066;
    factors[1][9] = .003679;

    factors[2][0] = .001000;
    factors[2][1] = .000750;
    factors[2][2] = .000500;
    factors[2][3] = .000250;
    factors[2][4] = .000000;
    factors[2][5] = -.000250;
    factors[2][6] = -.000500;
    factors[2][7] = -.000750;
    factors[2][8] = -.001000;
    factors[2][9] = -.001250;

    // setting up multiple swaptions
    swaptions = (parm *)malloc(sizeof(parm) * nSwaptions);

    for (int i = 0; i < nSwaptions; i++){
        swaptions[i].Id = i;
        swaptions[i].iN = iN;
        swaptions[i].iFactors = iFactors;
        swaptions[i].dYears = 5.0 + ((int)(60 * RanUnif(&seed))) * 0.25; // 5 to 20 years in 3 month intervals

        swaptions[i].dStrike = 0.1 + ((int)(49 * RanUnif(&seed))) * 0.1; // strikes ranging from 0.1 to 5.0 in steps of 0.1
        swaptions[i].dCompounding = 0;
        swaptions[i].dMaturity = 1.0;
        swaptions[i].dTenor = 2.0;
        swaptions[i].dPaymentInterval = 1.0;

        swaptions[i].pdYield = dvector(0, iN - 1);
        swaptions[i].pdYield[0] = .1;
        for (int j = 1; j <= swaptions[i].iN - 1; ++j)
            swaptions[i].pdYield[j] = swaptions[i].pdYield[j - 1] + .005;

        swaptions[i].ppdFactors = dmatrix(0, swaptions[i].iFactors - 1, 0, swaptions[i].iN - 2);
        for (int k = 0; k <= swaptions[i].iFactors - 1; ++k)
            for (int j = 0; j <= swaptions[i].iN - 2; ++j)
                swaptions[i].ppdFactors[k][j] = factors[k][j];
    }

    // KERNEL:
    if(vm["gpu"].as<bool>()){
        SwaptionsGpu::kernel(
            swaptions,
            nSwaptions,
            gpuBlockSize,
            nTrials, //we assume the number of trials is higher than the total gpu threads
            nTrials,
            swaption_seed
        );
    } else {
        SwaptionsCpu::kernel(
            swaptions,
            nSwaptions,
            cpuThreads,
            cpuTileSize,
            nTrials,
            swaption_seed
        );
    }
    
    
    // OUTPUT:
    Swaptions::WriteSwaptionsPrices(
        vm["output-file"].as<std::string>(),
        nTrials,
        nSwaptions,
        swaptions
    );

    // CLEANUP:
    for (int i = 0; i < nSwaptions; i++){
        free_dvector(swaptions[i].pdYield, 0, swaptions[i].iN - 1);
        free_dmatrix(swaptions[i].ppdFactors, 0, swaptions[i].iFactors - 1, 0, swaptions[i].iN - 2);
    }

    free(swaptions);
    free_dmatrix(factors, 0, iFactors - 1, 0, iN - 2);

    return 0;
}

po::options_description SetupOptions()
{
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "Display help message")
    ("gpu", po::bool_switch(), "use gpu")
    ("cpu-threads", po::value<unsigned int>()->default_value(1), "number of cpu threads")
    ("cpu-tile-exp", po::value<unsigned int>()->default_value(0), "tile exp; tile_size = 16*2^X")
    ("gpu-block-exp", po::value<unsigned int>()->default_value(0), "block exp; tile_size = 32*2^X")

    ("swaptions,N", po::value<unsigned int>(), "number of swaptions")
    ("trials,T", po::value<unsigned int>(), "number of trials")
    ("output-file,O", po::value<std::string>(), "output result file")
    ;

    return desc;
}
