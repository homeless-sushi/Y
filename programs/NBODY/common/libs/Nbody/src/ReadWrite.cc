#include "Nbody/Body.h"
#include "Nbody/ReadWrite.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace Nbody
{
    void ReadBodyFile(
        const std::string& fileURL,
        std::vector<Body>& bodies, 
        float& simTime,
        float& timeStep
    )
    {
        std::ifstream bodyFile(fileURL);
        if (!bodyFile.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }

        std::string lineString;
        getline(bodyFile, lineString);
        std::istringstream timeLineStream(lineString);
        timeLineStream >> simTime;
        timeLineStream >> timeStep;

        while(bodyFile.good()){

            getline(bodyFile, lineString);

            if(lineString.empty())
                break;

            std::istringstream bodyLineStream(lineString);
            Body body;
            bodyLineStream >> body.pos.x;
            bodyLineStream >> body.pos.y;
            bodyLineStream >> body.pos.z;
            bodyLineStream >> body.vel.x;
            bodyLineStream >> body.vel.y;
            bodyLineStream >> body.vel.z;

            bodies.push_back(body);
        }
    }

    void WriteBodyFile(
        const std::string fileURL,
        const std::vector<Body>& bodies,
        float simTime,
        float timeStep
    )
    {
	    std::ofstream outFile(fileURL);
        if (!outFile.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }

        outFile << std::fixed << std::setprecision(6);
        outFile << simTime << " " << timeStep << std::endl;
        for (const auto& body : bodies){
            outFile << body.pos.x << " " << body.pos.y << " " << body.pos.z
                << " " << body.vel.x << " " << body.vel.y << " " << body.vel.z << std::endl;
        }
    }

    void WriteCSVFile(
        const std::string fileURL,
        const std::vector<Body>& bodies,
        float targetSimTime,
        float targetTimeStep,
        float actualSimTime,
        float actualTimeStep,
        int precision
    )
    {
	    std::ofstream outFile(fileURL);
        if (!outFile.is_open()){
            throw std::runtime_error("Cannot open file: " + fileURL);
        }

        outFile << std::fixed << std::setprecision(6);
        outFile << "# " << "target simulation time: " << targetSimTime << std::endl;
        outFile << "# " << "target time step: " << targetTimeStep << std::endl;
        outFile << "# " << "actual simulation time: " << actualSimTime << std::endl;
        outFile << "# " << "actual time step: " << actualTimeStep << std::endl;
        outFile << "# " << "precision: " << precision << std::endl;
        outFile << "X, Y, Z, VX, VY, VZ" << std::endl;
        for (const auto& body : bodies){
            outFile << body.pos.x << ", " << body.pos.y << ", " << body.pos.z
                << ", " << body.vel.x << ", " << body.vel.y << ", " << body.vel.z << std::endl;
        }
    }
}
