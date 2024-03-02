#ifndef HISTO_HISTO_HISTO_H
#define HISTO_HISTO_HISTO_H

#include <vector>

namespace Histo 
{
    class Histo
    {
        public:
            Histo(std::vector<unsigned short> rgb);
            virtual void run() = 0;

            virtual std::vector<unsigned> getResult() = 0;

            virtual ~Histo() = default;

        protected:
            std::vector<unsigned short> rgb;
            std::vector<unsigned> histo;
    };
}

#endif //HISTO_HISTO_HISTO_H
