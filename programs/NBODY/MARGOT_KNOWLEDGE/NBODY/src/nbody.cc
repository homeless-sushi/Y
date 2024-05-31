#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <csignal>

#include <boost/program_options.hpp>

#include "Nbody/Body.h"
#include "Nbody/ReadWrite.h"
#include "Nbody/NbodyCpu.h"
#include "Nbody/NbodyCuda.h"

#include "Knobs/Device.h"
#include "Knobs/Precision.h"

#include "AppRegisterServer/Frequency.h"
#include "AppRegisterServer/Sensors.h"

typedef std::result_of<decltype(&std::chrono::system_clock::now)()>::type TimePoint;
typedef std::chrono::duration<double, std::milli> Duration;

Sensors::Sensors sensors; 

void CastKnobs(unsigned int gpuBlockSizeExp, Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize);
int kernel(unsigned int inputSize, Knobs::DEVICE device, unsigned int cpuThreads, unsigned int gpuBlockSizeExp, unsigned int precision);

int main(int argc, char *argv[])
{
    std::cout << "INPUT_SIZE,F_CPU,F_GPU,DEVICE_ID,N_THREADS,GPU_BLOCK_SIZE_EXP,PRECISION,TIME,KERNEL_FRACTION,GPU_W" << std::endl;

    std::vector<unsigned int> inputSizes{256,512,1024,2048,4096};
    const unsigned int precisionMin = 0;
    const unsigned int precisionMax = 100;

    Frequency::CPU_FRQ cpuFreq = Frequency::getMinCpuFreq();
    Frequency::GPU_FRQ gpuFreq = Frequency::getMinGpuFreq();

    const unsigned int gpuBlockSizeExpMin = 0;
    const unsigned int gpuBlockSizeExpMax = 5;

    const unsigned int cpuThreadsMin = 1;
    const unsigned int cpuThreadsMax= 4;

    // INPUT
    for(const unsigned int inputSize : inputSizes){
        // DEVICE
        for (Knobs::DEVICE device : {Knobs::DEVICE::CPU,Knobs::DEVICE::GPU}){
            // COOL THE DEVICE
            if(device == Knobs::DEVICE::CPU){
                Frequency::CPU_FRQ cpuFreq = Frequency::getMinCpuFreq();
                Frequency::GPU_FRQ gpuFreq = Frequency::getMinGpuFreq();
                Frequency::SetGpuFreq(gpuFreq);
                while(true){
                
                    Frequency::SetCpuFreq(cpuFreq);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

                    for(unsigned int cpuThreads = cpuThreadsMin; cpuThreads <= cpuThreadsMax; cpuThreads++){
                        for(unsigned int precision = precisionMin; precision <= precisionMax; precision+=10){
                            const unsigned int gpuBlockSizeExp = gpuBlockSizeExpMin;
                            std::cout << inputSize << "," << static_cast<unsigned int>(cpuFreq) << "," << static_cast<unsigned int>(gpuFreq) << ",";
                            kernel(inputSize, device, cpuThreads, gpuBlockSizeExp, precision);//
                            std::cout << std::endl;
                        }
                    }

                    if(cpuFreq == Frequency::getMaxCpuFreq()){ break; }
                    cpuFreq = Frequency::getNextCpuFreq(cpuFreq);
                }
            } else if (device == Knobs::DEVICE::GPU) {
                Frequency::CPU_FRQ cpuFreq = Frequency::getMinCpuFreq();
                while(true){
                
                    Frequency::SetCpuFreq(cpuFreq);
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));

                    Frequency::GPU_FRQ gpuFreq = Frequency::getMinGpuFreq();
                    while(true){
                    
                        Frequency::SetGpuFreq(gpuFreq);
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));

                        for(unsigned int gpuBlockSizeExp = gpuBlockSizeExpMin; gpuBlockSizeExp <= gpuBlockSizeExpMax; gpuBlockSizeExp++){
                            for(unsigned int precision = precisionMin; precision <= precisionMax; precision+=10){
                                
                                //repeat for 1 sec for gpu power
                                TimePoint startGpu, stopGpu;
                                Duration currDuration = std::chrono::duration<double, std::milli>(0);
                                Duration maxDuration = std::chrono::duration<double, std::milli>(1000);
                                while(currDuration < maxDuration){
                                    startGpu = std::chrono::system_clock::now();
                                    const unsigned int cpuThreads = cpuThreadsMin;
                                    std::cout << inputSize << "," << static_cast<unsigned int>(cpuFreq) << "," << static_cast<unsigned int>(gpuFreq) << ",";
                                    kernel(inputSize, device, cpuThreads, gpuBlockSizeExp, precision);//
                                    std::cout << std::endl;
                                    stopGpu = std::chrono::system_clock::now();
                                    currDuration += std::chrono::duration<double, std::milli>((stopGpu - startGpu));
                                }
                            }
                        }

                        if(gpuFreq == Frequency::getMaxGpuFreq()){ break; }
                        gpuFreq = Frequency::getNextGpuFreq(gpuFreq);
                    }

                    if(cpuFreq == Frequency::getMaxCpuFreq()){ break; }
                    cpuFreq = Frequency::getNextCpuFreq(cpuFreq);
                }
            }
        }
    }
    
    return 0;
}

void CastKnobs(
    unsigned int gpuBlockSizeExp,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize
)
{
    gpuBlockSize = static_cast<Knobs::GpuKnobs::BLOCK_SIZE>(
        Knobs::GpuKnobs::BLOCK_32 << gpuBlockSizeExp
    );
}

int kernel(
    unsigned int inputSize,
    Knobs::DEVICE device,
    unsigned int cpuThreads,
    unsigned int gpuBlockSizeExp, 
    unsigned int precision
)
{
    TimePoint startLoop, stopLoop;
    TimePoint startKernel, stopKernel;

    std::string inputFileURL = 
        "/home/miele/Vivian/Thesis/apps/Y/programs/NBODY/data/in/" 
        + std::to_string(inputSize)
        + ".txt";
    std::string outputFileURL = 
        "/home/miele/Vivian/Thesis/apps/Y/programs/NBODY/data/out/" 
        + std::to_string(inputSize)
        + ".txt";

    //START: LOOP
    startLoop = std::chrono::system_clock::now();

    Knobs::GpuKnobs::BLOCK_SIZE gpuBlockSize;
    CastKnobs(
        gpuBlockSizeExp,
        gpuBlockSize
    );

    std::vector<Nbody::Body> bodies;
    float targetSimulationTime;
    float targetTimeStep;
    Nbody::ReadBodyFile(inputFileURL, bodies, targetSimulationTime, targetTimeStep);
    float actualTimeStep = targetTimeStep;
    float approximateSimulationTime = Knobs::GetApproximateSimTime(
        targetSimulationTime, 
        targetTimeStep,
        precision
    );

    std::unique_ptr<Nbody::Nbody> nbody( 
        device == Knobs::DEVICE::GPU ?
        static_cast<Nbody::Nbody*>(new NbodyCuda::NbodyCuda(bodies, approximateSimulationTime, actualTimeStep, gpuBlockSize)) :
        static_cast<Nbody::Nbody*>(new NbodyCpu::NbodyCpu(bodies, approximateSimulationTime, actualTimeStep, cpuThreads))
    );

    //START: KERNEL
    startKernel= std::chrono::system_clock::now();
    nbody->run();
    stopKernel = std::chrono::system_clock::now();
    //STOP: KERNEL
    
    bodies = nbody->getResult();
    Nbody::WriteBodyFile(outputFileURL, 
        bodies,
        nbody->getSimulatedTime(),
        actualTimeStep
    );

    // STOP: LOOP
    stopLoop = std::chrono::system_clock::now();

    Duration loopDuration = std::chrono::duration<double, std::milli>((stopLoop - startLoop));
    Duration kernelDuration = std::chrono::duration<double, std::milli>((stopKernel - startKernel));
    const float kernelFraction = kernelDuration/loopDuration;
    sensors.readSensors();
    float gpuPow = sensors.getGpuW();

    //write knobs and metrics
    std::cout <<
        static_cast<unsigned int>(device) << "," <<
        cpuThreads << "," <<
        gpuBlockSizeExp << "," <<
        precision << "," <<
        loopDuration.count() << "," <<
        kernelFraction << "," <<
        gpuPow;

    return 0;
}