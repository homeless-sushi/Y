#ifndef HISTO_HISTO_HISTO_CPU_H
#define HISTO_HISTO_HISTO_CPU_H

#include <Histo/Histo.h>

#include <vector>

namespace Histo 
{
    class HistoCpu : public ::Histo::Histo
    {
        public:
            HistoCpu(
                std::vector<unsigned short> rgb,
                unsigned nThreads
            );
            void run();

            std::vector<unsigned> getResult();

        private:
            unsigned nThreads;
    };
}

#endif //HISTO_HISTO_HISTO_CPU_H
