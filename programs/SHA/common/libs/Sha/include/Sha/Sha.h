#ifndef SHA_SHA_SHA_H
#define SHA_SHA_SHA_H

#include <iostream>

namespace Sha
{
    class ShaInfo
    {
        private:
            unsigned long digest[5];
            unsigned long countLow = 0UL;
            unsigned long countHigh = 0UL;
            unsigned long data[16];	

            void transform();
            void update(unsigned char* buffer, unsigned size);
            void final();

            friend std::ostream& operator<<(std::ostream& os, const ShaInfo& sha);

        public:
            ShaInfo(unsigned long seed);
            
            void digestFile(std::string fileUrl);
    };

    void byteReverse(unsigned long *buffer, unsigned size);
}

#endif //SHA_SHA_SHA_H