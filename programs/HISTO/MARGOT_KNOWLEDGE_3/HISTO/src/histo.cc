#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <csignal>

#include "Histo/Histo.h"
#include "Histo/HistoCuda.h"
#include "Histo/HistoCpu.h"
#include "Histo/ReadWrite.h"

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

void CastKnobs(unsigned int gpuNBlocksExp, unsigned int gpuBlockSizeExp, Knobs::GpuKnobs::N_BLOCKS& gpuNBlocks, Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize);
int kernel(
    app_data* data,
    int dataSemId,
    unsigned int inputSize, 
    Knobs::DEVICE device, 
    Frequency::CPU_FRQ cpuFreq,
    Frequency::GPU_FRQ gpuFreq,
    unsigned int cpuThreads, 
    unsigned int gpuNBlocksExp, 
    unsigned int gpuBlockSizeExp
);

int main(int argc, char *argv[])
{
    margot::init();

    //Attach to controller
    struct app_data* data = registerAttach(
        "HISTO,480",
        "HISTO", 480,
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

    std::cout << "INPUT_SIZE,F_CPU,F_GPU,DEVICE_ID,N_THREADS,GPU_N_BLOCKS_EXP,GPU_BLOCK_SIZE_EXP,TIME,KERNEL_FRACTION,GPU_W" << std::endl;

    std::vector<unsigned int> inputSizes{480,720,1080};

    Frequency::CPU_FRQ cpuFreq = Frequency::getMinCpuFreq();
    Frequency::GPU_FRQ gpuFreq = Frequency::getMinGpuFreq();

    const unsigned int gpuNBlocksExpMin = 0;
    const unsigned int gpuNBlocksExpMax = 5;
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
                
                while(true){
                
                    Frequency::SetCpuFreq(cpuFreq);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

                    for(unsigned int cpuThreads = cpuThreadsMin; cpuThreads <= cpuThreadsMax; cpuThreads++){
                        const unsigned int gpuNBlocksExp = gpuNBlocksExpMin;
                        const unsigned int gpuBlockSizeExp = gpuBlockSizeExpMin;
                        std::cout << inputSize << "," << static_cast<unsigned int>(cpuFreq) << "," << static_cast<unsigned int>(gpuFreq) << ",";
                        kernel(data, dataSemId, inputSize, device, cpuFreq, gpuFreq, cpuThreads, gpuNBlocksExp, gpuBlockSizeExp);//
                        std::cout << std::endl;
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
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

                        for(unsigned int gpuNBlocksExp = gpuNBlocksExpMin; gpuNBlocksExp <= gpuNBlocksExpMax; gpuNBlocksExp++){
                            for(unsigned int gpuBlockSizeExp = gpuBlockSizeExpMin; gpuBlockSizeExp <= gpuBlockSizeExpMax; gpuBlockSizeExp++){
                                //repeat for 1 sec for gpu power
                                TimePoint startGpu, stopGpu;
                                Duration currDuration = std::chrono::duration<double, std::milli>(0);
                                Duration maxDuration = std::chrono::duration<double, std::milli>(1000);
                                while(currDuration < maxDuration){
                                    startGpu = std::chrono::system_clock::now();
                                    
                                    const unsigned int cpuThreads = cpuThreadsMin;
                                    std::cout << inputSize << "," << static_cast<unsigned int>(cpuFreq) << "," << static_cast<unsigned int>(gpuFreq) << ",";
                                    kernel(data, dataSemId, inputSize, device, cpuFreq, gpuFreq, cpuThreads, gpuNBlocksExp, gpuBlockSizeExp);//
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
    unsigned int gpuNBlocksExp,
    unsigned int gpuBlockSizeExp,
    Knobs::GpuKnobs::N_BLOCKS& gpuNBlocks,
    Knobs::GpuKnobs::BLOCK_SIZE& gpuBlockSize
)
{
    gpuNBlocks = Knobs::nBlocksfromExponent(gpuNBlocksExp);
    gpuBlockSize = Knobs::blockSizefromExponent(gpuBlockSizeExp);
}

int kernel(
    app_data* data,
    int dataSemId,
    unsigned int inputSize, 
    Knobs::DEVICE device,
    Frequency::CPU_FRQ cpuFreq,
    Frequency::GPU_FRQ gpuFreq,
    unsigned int cpuThreads, 
    unsigned int gpuNBlocksExp, 
    unsigned int gpuBlockSizeExp
)
{
    TimePoint startLoop, stopLoop;
    TimePoint startKernel, stopKernel;
    bool error = false;

    std::string inputFileURL = 
        "/home/miele/Vivian/Thesis/apps/Y/programs/HISTO/data/in/" 
        + std::to_string(inputSize)
        + ".bin";
    std::string outputFileURL = 
        "/home/miele/Vivian/Thesis/apps/Y/programs/HISTO/data/out/" 
        + std::to_string(inputSize)
        + ".bin";

    bool useGpu = (device == Knobs::DEVICE::GPU);
    Knobs::GpuKnobs::N_BLOCKS gpuNBlocks;
    Knobs::GpuKnobs::BLOCK_SIZE gpuBlockSize;
    CastKnobs(
        gpuNBlocksExp,
        gpuBlockSizeExp,
        gpuNBlocks,
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
    unsigned int ignore6 = gpuNBlocksExp;
    Knobs::GpuKnobs::N_BLOCKS ignore7;
    Knobs::GpuKnobs::BLOCK_SIZE ignore8;
    if(margot::histo::update(useGpu, cpuFreq, gpuFreq, inputSize, cpuThreads, ignore5, ignore6)){   
        CastKnobs(
            ignore6,
            ignore5,
            ignore7,
            ignore8
        );
        margot::histo::context().manager.configuration_applied(); 
    }
    margot::histo::start_monitors();
    //STOP: MARGOT

    //START: WINDUP
    unsigned imgWidth, imgHeight;
    std::vector<unsigned short> rgb;
    Histo::ReadBinaryDataFile(
        inputFileURL,
        imgWidth, imgHeight,
        rgb
    );

    std::unique_ptr<Histo::Histo> histo(
        device == Knobs::DEVICE::GPU ?
        static_cast<Histo::Histo*>(new Histo::HistoCuda(
            rgb,
            gpuNBlocks, gpuBlockSize
        )) :
        static_cast<Histo::Histo*>(new Histo::HistoCpu(
            rgb,
            cpuThreads
        ))
    );
    //STOP: WINDUP

    //START: KERNEL
    startKernel= std::chrono::system_clock::now();
    histo->run();
    stopKernel = std::chrono::system_clock::now();
    //STOP: KERNEL
    
    //START: ENDING
    std::vector<unsigned> res = histo->getResult();
    Histo::WriteHistogramFile(outputFileURL, res);
    margot::histo::stop_monitors();
    //STOP: ENDING

    //START: CONTROLLER PUSH
    autosleep(data, 100000);
    error = binarySemaphoreWait(dataSemId);
    addTick(data, 1);
    error = binarySemaphorePost(dataSemId);
    //STOP: CONTROLLER PUSH

    stopLoop = std::chrono::system_clock::now();
    //STOP: LOOP

    Duration kernelDuration = std::chrono::duration<double, std::milli>((stopKernel - startKernel));
    Duration loopDuration = std::chrono::duration<double, std::milli>((stopLoop - startLoop));
    const float kernelFraction = kernelDuration/loopDuration;
    sensors.readSensors();
    float gpuPow = sensors.getGpuW();

    //write knobs and metrics
    std::cout <<
        static_cast<unsigned int>(device) << "," <<
        cpuThreads << "," <<
        gpuNBlocksExp << "," <<
        gpuBlockSizeExp << "," <<
        loopDuration.count() << "," <<
        kernelFraction << "," <<
        gpuPow;

    return 0;
}
