#ifndef DUMMY_DUMMY_READWRITE_H
#define DUMMY_DUMMY_READWRITE_H

#include <string>
#include <vector>

namespace Dummy
{
    void ReadFile(
        const std::string& fileURL,
        std::vector<float>& data
    );
        
    void WriteFile(
        const std::string fileURL,
        const std::vector<float>& data
    );
}

#endif //DUMMY_DUMMY_READWRITE_H