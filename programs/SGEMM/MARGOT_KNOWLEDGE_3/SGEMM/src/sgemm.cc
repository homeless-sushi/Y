#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <csignal>

#include "Sgemm/Matrix.h"
#include "Sgemm/Sgemm.h"
#include "Sgemm/SgemmCpu.h"
#include "Sgemm/SgemmCuda.h"
#include "Sgemm/ReadWrite.h"

#include "Knobs/Device.h"

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

void CastKnobs(unsigned int cpuTileSizeExp, unsigned int gpuTileSizeExp, unsigned int& cpuTileSize, Knobs::GpuKnobs::TILE_SIZE& gpuTileSize);
int kernel(
    app_data* data,
    int dataSemId,
    unsigned int inputSize,
    Knobs::DEVICE device,
    Frequency::CPU_FRQ cpuFreq,
    Frequency::GPU_FRQ gpuFreq,
    unsigned int cpuThreads,
    unsigned int cpuTileSizeExp, 
    unsigned int gpuTileSizeExp
);

int main(int argc, char *argv[])
{
    margot::init();

    struct app_data* data = registerAttach(
        "SGEMM,32",
        "SGEMM", 32,
        0.001,
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

    std::cout << "INPUT_SIZE,F_CPU,F_GPU,DEVICE_ID,N_THREADS,CPU_TILE_SIZE_EXP,GPU_TILE_SIZE_EXP,TIME,KERNEL_FRACTION,GPU_W" << std::endl;

    std::vector<unsigned int> inputSizes{32,64,128,256};

    Frequency::CPU_FRQ cpuFreq = Frequency::getMinCpuFreq();
    Frequency::GPU_FRQ gpuFreq = Frequency::getMinGpuFreq();

    const unsigned int gpuTileSizeExpMin = 0;
    const unsigned int gpuTileSizeExpMax = 1;

    const unsigned int cpuThreadsMin = 1;
    const unsigned int cpuThreadsMax= 4;
    const unsigned int cpuTileSizeExpMin = 0;
    const unsigned int cpuTileSizeExpMax = 1;

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
                        for(unsigned int cpuTileSizeExp = cpuTileSizeExpMin; cpuTileSizeExp <= cpuTileSizeExpMax; cpuTileSizeExp++){
                            const unsigned int gpuTileSizeExp = gpuTileSizeExpMin;
                            std::cout << inputSize << "," << static_cast<unsigned int>(cpuFreq) << "," << static_cast<unsigned int>(gpuFreq) << ",";
                            kernel(data, dataSemId, inputSize, device, cpuFreq, gpuFreq, cpuThreads, cpuTileSizeExp, gpuTileSizeExp);//
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
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

                    Frequency::GPU_FRQ gpuFreq = Frequency::getMinGpuFreq();
                    while(true){
                    
                        Frequency::SetGpuFreq(gpuFreq);
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));

                        for(unsigned int gpuTileSizeExp = gpuTileSizeExpMin; gpuTileSizeExp <= gpuTileSizeExpMax; gpuTileSizeExp++){
                                
                            //repeat for 1 sec for gpu power
                            TimePoint startGpu, stopGpu;
                            Duration currDuration = std::chrono::duration<double, std::milli>(0);
                            Duration maxDuration = std::chrono::duration<double, std::milli>(1000);
                            while(currDuration < maxDuration){
                                startGpu = std::chrono::system_clock::now();
                                
                                const unsigned int cpuThreads = cpuThreadsMin;
                                const unsigned int cpuTileSizeExp = cpuTileSizeExpMin;
                                std::cout << inputSize << "," << static_cast<unsigned int>(cpuFreq) << "," << static_cast<unsigned int>(gpuFreq) << ",";
                                kernel(data, dataSemId, inputSize, device, cpuFreq, gpuFreq, cpuThreads, cpuTileSizeExp, gpuTileSizeExp);//
                                std::cout << std::endl;
                                stopGpu = std::chrono::system_clock::now();
                                currDuration += std::chrono::duration<double, std::milli>((stopGpu - startGpu));
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
    unsigned int cpuTileSizeExp,
    unsigned int gpuTileSizeExp,
    unsigned int& cpuTileSize,
    Knobs::GpuKnobs::TILE_SIZE& gpuTileSize
)
{
    cpuTileSize = Knobs::GetCpuTileSizeFromExponent(cpuTileSizeExp);
    gpuTileSize = Knobs::GetGpuTileSizeFromExponent(gpuTileSizeExp);
}

int kernel(
    app_data* data,
    int dataSemId,
    unsigned int inputSize,
    Knobs::DEVICE device,
    Frequency::CPU_FRQ cpuFreq,
    Frequency::GPU_FRQ gpuFreq,
    unsigned int cpuThreads,
    unsigned int cpuTileSizeExp, 
    unsigned int gpuTileSizeExp
)
{
    TimePoint startLoop, stopLoop;
    TimePoint startKernel, stopKernel;
    bool error = false;

    std::string inputFileURL = 
        "/home/miele/Vivian/Thesis/apps/Y/programs/SGEMM/data/in/" 
        + std::to_string(inputSize) + "x" + std::to_string(inputSize)
        + ".txt";
    std::string outputFileURL = 
        "/home/miele/Vivian/Thesis/apps/Y/programs/SGEMM/data/out/" 
        + std::to_string(inputSize) + "x" + std::to_string(inputSize)
        + ".txt";

    bool useGpu = (device == Knobs::DEVICE::GPU);
    unsigned int cpuTileSize;
    Knobs::GpuKnobs::TILE_SIZE gpuTileSize;
    CastKnobs(
        cpuTileSizeExp,
        gpuTileSizeExp,
        cpuTileSize,
        gpuTileSize
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
    unsigned int ignore5 = cpuTileSizeExp; 
    unsigned int ignore6 = gpuTileSizeExp;
    unsigned int ignore7;
    Knobs::GpuKnobs::TILE_SIZE ignore8;
    if(margot::sgemm::update(useGpu, cpuFreq, gpuFreq, inputSize, cpuThreads, ignore5, ignore6)){   
        CastKnobs(
            ignore5,
            ignore6,
            ignore7,
            ignore8
        );
        margot::sgemm::context().manager.configuration_applied(); 
    }
    margot::sgemm::start_monitors();
    //STOP: MARGOT

    //START: WINDUP
    Sgemm::Matrix a(Sgemm::ReadMatrixFile(inputFileURL));
    Sgemm::Matrix b(Sgemm::ReadMatrixFile(inputFileURL));
    Sgemm::Matrix c(Sgemm::ReadMatrixFile(inputFileURL));

    std::unique_ptr<Sgemm::Sgemm> sgemm( 
        device == Knobs::DEVICE::GPU ?
        static_cast<Sgemm::Sgemm*>(new Sgemm::SgemmCuda(1,1,a,b,c,gpuTileSize)) :
        static_cast<Sgemm::Sgemm*>(new Sgemm::SgemmCpu(1,1,a,b,c,cpuThreads,cpuTileSize))
    );
    //STOP: WINDUP

    //START: KERNEL
    startKernel= std::chrono::system_clock::now();
    sgemm->run();
    stopKernel = std::chrono::system_clock::now();
    //STOP: KERNEL
    
    //START: ENDING
    Sgemm::Matrix res(sgemm->getResult());
    Sgemm::WriteMatrixFile(outputFileURL, res);
    margot::sgemm::stop_monitors();
    //STOP: ENDING

    //START: CONTROLLER PUSH
    autosleep(data, 100000);
    error = binarySemaphoreWait(dataSemId);
    addTick(data, 1);
    error = binarySemaphorePost(dataSemId);
    //STOP: CONTROLLER PUSH

    stopLoop = std::chrono::system_clock::now();
    //STOP: LOOP

    Duration loopDuration = std::chrono::duration<double, std::milli>((stopLoop - startLoop));
    Duration kernelDuration = std::chrono::duration<double, std::milli>((stopKernel - startKernel));
    const float kernelFraction = kernelDuration/loopDuration;
    sensors.readSensors();
    float gpuPow = sensors.getGpuW();

    //write knobs and metrics
    std::cout <<
        static_cast<unsigned int>(device) << "," <<
        cpuThreads << "," <<
        cpuTileSizeExp << "," <<
        gpuTileSizeExp << "," <<
        loopDuration.count() << "," <<
        kernelFraction << "," <<
        gpuPow;

    return 0;
}
