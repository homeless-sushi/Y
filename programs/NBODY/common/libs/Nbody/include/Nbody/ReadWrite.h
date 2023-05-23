#ifndef NBODY_NBODY_READWRITE_H
#define NBODY_NBODY_READWRITE_H

#include "Nbody/Body.h"

#include <string>
#include <vector>

namespace Nbody
{
    void ReadBodyFile(
        const std::string& fileURL,
        std::vector<Body>& bodies, 
        float& simTime,
        float& timeStep
    );
        
    void WriteBodyFile(
        const std::string fileURL,
        const std::vector<Body>& bodies,
        float simTime,
        float timeStep
    );

    void WriteCSVFile(
        const std::string fileURL,
        const std::vector<Body>& bodies,
        float targetSimTime,
        float targetTimeStep,
        float actualSimTime,
        float actualTimeStep,
        int precision
    );
}

#endif //NBODY_NBODY_READWRITE_H