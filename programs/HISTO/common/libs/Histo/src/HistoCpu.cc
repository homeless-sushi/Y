#include <Histo/Const.h>
#include <Histo/Histo.h>
#include <Histo/HistoCpu.h>

#include <vector>
#include <iostream>

#include <cstring>

namespace Histo 
{
    HistoCpu::HistoCpu(
        std::vector<unsigned short> rgb,
        unsigned nThreads
    ) :
        ::Histo::Histo{rgb},
        nThreads{nThreads}
    {};

    void HistoCpu::run()
    {
        #pragma omp parallel \
        num_threads(nThreads)
        {
            std::vector<unsigned> local(N_CHANNEL_VALUES*N_CHANNELS);
            //add size of workload
            #pragma omp for
            for (unsigned i = 0; i < rgb.size(); i+=N_CHANNELS) {
                #pragma unroll
                for(unsigned j = 0; j < N_CHANNELS; ++j){
                    ++local[N_CHANNEL_VALUES*j + rgb[i+j]];
                }
            }

            #pragma omp critical
            {
                #pragma unroll
                for(unsigned i = 0; i < N_CHANNEL_VALUES*N_CHANNELS; ++i){
                    histo[i]+=local[i];
                }
            }
        }
    };

    std::vector<unsigned> HistoCpu::getResult(){ return histo; };
}
