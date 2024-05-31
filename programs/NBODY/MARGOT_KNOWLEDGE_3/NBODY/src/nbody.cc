#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <csignal>


#include "Nbody/Body.h"
#include "Nbody/ReadWrite.h"
#include "Nbody/NbodyCpu.h"
#include "Nbody/NbodyCuda.h"

#include "Knobs/Device.h"
#include "Knobs/Precision.h"

#include <boost/program_options.hpp>

#include "AppRegisterServer/Frequency.h"
#include "AppRegisterServer/Sensors.h"

#include <boost/program_options.hpp>

#include "AppRegisterServer/Frequency.h"
#include "AppRegisterServer/Sensors.h"

#include "AppRegisterCommon/AppRegister.h"
#include "AppRegisterCommon/Semaphore.h"

#include "AppRegisterClient/AppRegister.h"
#include "AppRegisterClient/Utils.h"

#include <margot/margot.hpp>

typedef std::result_of<decltype(&std::chrono::system_clock::now)()>::type TimePoint;
typedef std::chrono::duration<double, std::milli> Duration;

Sensors::Sensors sensors; 

void CastKnobs(unsigned int gpuBlockSizeExp, Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize);
int kernel(
    app_data* data,
    int dataSemId,
    unsigned int inputSize,
    Knobs::DEVICE device,
    Frequency::CPU_FRQ cpuFreq,
    Frequency::GPU_FRQ gpuFreq,
    unsigned int cpuThreads,
    unsigned int gpuBlockSizeExp,
    unsigned int precision
);

int main(int argc, char *argv[])
{
    margot::init();

    //Attach to controller
    struct app_data* data = registerAttach(
        "NBODY,256",
        "NBODY", 256,
        0.01,
        4,
        true
    );
    int dataSemId = semget(getpid(), 1, 0);
    while(true){
        if(isRegistered(data)){
            setTickStartTime(data);
            break;
        }
    }

    std::cout << "INPUT_SIZE,F_CPU,F_GPU,DEVICE_ID,N_THREADS,GPU_BLOCK_SIZE_EXP,PRECISION,TIME,KERNEL_FRACTION,GPU_W" << std::endl;

    std::vector<unsigned int> inputSizes{256};
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
                            kernel(data, dataSemId, inputSize, device, cpuFreq, gpuFreq, cpuThreads, gpuBlockSizeExp, precision);//
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
                                    kernel(data, dataSemId, inputSize, device, cpuFreq, gpuFreq, cpuThreads, gpuBlockSizeExp, precision);//
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

    registerDetach(data);

    Frequency::SetCpuFreq(Frequency::getMinCpuFreq());
    Frequency::SetGpuFreq(Frequency::getMinGpuFreq());
    
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
    app_data* data,
    int dataSemId,
    unsigned int inputSize,
    Knobs::DEVICE device,
    Frequency::CPU_FRQ cpuFreq,
    Frequency::GPU_FRQ gpuFreq,
    unsigned int cpuThreads,
    unsigned int gpuBlockSizeExp, 
    unsigned int precision
)
{
    TimePoint startLoop, stopLoop;
    TimePoint startKernel, stopKernel;
    bool error = false;

    std::string inputFileURL = 
        "/home/miele/Vivian/Thesis/apps/Y/programs/NBODY/data/in/" 
        + std::to_string(inputSize)
        + ".txt";
    std::string outputFileURL = 
        "/home/miele/Vivian/Thesis/apps/Y/programs/NBODY/data/out/" 
        + std::to_string(inputSize)
        + ".txt";

    bool useGpu = (device == Knobs::DEVICE::GPU);
    Knobs::GpuKnobs::BLOCK_SIZE gpuBlockSize;
    CastKnobs(
        gpuBlockSizeExp,
        gpuBlockSize
    );
        
    //START: LOOP
    startLoop = std::chrono::system_clock::now();

    //START: CONTROLLER PULL
    error = binarySemaphoreWait(dataSemId);
    const int ignore1 = getNCpuCores(data);
    const bool ignore2 = getUseGpu(data) ? Knobs::DEVICE::GPU : Knobs::DEVICE::CPU;
    const Frequency::CPU_FRQ ignore3 = static_cast<Frequency::CPU_FRQ>(getCpuFreq(data));
    const Frequency::GPU_FRQ ignore4 = static_cast<Frequency::GPU_FRQ>(getGpuFreq(data));
    error = binarySemaphorePost(dataSemId);
    //STOP: CONTROLLER PULL

    //START: MARGOT
    unsigned int ignore5 = gpuBlockSizeExp;
    unsigned int ignore6 = precision;
    Knobs::GpuKnobs::BLOCK_SIZE ignore7;
    if(margot::nbody::update(useGpu, cpuFreq, gpuFreq, inputSize, cpuThreads, ignore5, ignore6)){   
        CastKnobs(
            ignore5,
            ignore7
        );
        margot::nbody::context().manager.configuration_applied(); 
    }
    margot::nbody::start_monitors();
    //STOP: MARGOT

    //START: WINDUP
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
    //STOP: WINDUP

    //START: KERNEL
    startKernel= std::chrono::system_clock::now();
    nbody->run();
    stopKernel = std::chrono::system_clock::now();
    //STOP: KERNEL
    
    //START: ENDING
    bodies = nbody->getResult();
    Nbody::WriteBodyFile(outputFileURL, 
        bodies,
        nbody->getSimulatedTime(),
        actualTimeStep
    );
    margot::nbody::stop_monitors();
    margot::nbody::push_custom_monitor_values(precision);
    //STOP: ENDING

    //START: CONTROLLER PUSH
    autosleep(data, 100000);
    error = binarySemaphoreWait(dataSemId);
    addTick(data, 1);
    error = binarySemaphorePost(dataSemId);
    //STOP: CONTROLLER PUSH

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
