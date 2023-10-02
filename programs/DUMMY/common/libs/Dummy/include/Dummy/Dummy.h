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
            void runInfinite();

            std::vector<float> getResult();

            float getDataUploadTime() { return dataUploadTime; }
            float getKernelTime() { return kernelTime; }
            float getDataDownloadTime() { return dataDownloadTime; }

        private:
            std::vector<float> data;
            float* gpu_in;
            float* gpu_out;

            unsigned int gridLen;
            unsigned int blockLen;
            unsigned int times;

            float dataUploadTime = 0;
            float kernelTime = 0;
            float dataDownloadTime = 0;
    };
}

#endif //DUMMY_DUMMY_DUMMY_H
