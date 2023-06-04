#ifndef DUMMY_DUMMY_DUMMY_H
#define DUMMY_DUMMY_DUMMY_H

#include <vector>

namespace Dummy 
{
    class Dummy
    {
        public:
            Dummy(
                std::vector<float> data,
                unsigned int gridLen,
                unsigned int blockLen,
                unsigned int times
            );
            ~Dummy();
            void run();

            std::vector<float> getResult();

        private:
            std::vector<float> data;
            float* gpu_in;
            float* gpu_out;

            unsigned int gridLen;
            unsigned int blockLen;
            unsigned int times;
    };
}

#endif //DUMMY_DUMMY_DUMMY_H
