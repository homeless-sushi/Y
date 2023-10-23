#ifndef HISTO_HISTO_READWRITE_H
#define HISTO_HISTO_READWRITE_H

#include <string>
#include <vector>

namespace Histo
{
    void ReadDataFile(
        const std::string& fileURL,
        unsigned& imgWidth, unsigned& imgHeight,
        std::vector<unsigned short>& rgb
    );

    void ReadBinaryDataFile(
        const std::string& fileURL,
        unsigned& imgWidth, unsigned& imgHeight,
        std::vector<unsigned short>& rgb
    );

    void WriteHistogramFile(
        const std::string fileURL,
        const std::vector<unsigned>& histo
    );
}

#endif //HISTO_HISTO_READWRITE_H