#ifndef COMMON_DVFS_DVFS_H
#define COMMON_DVFS_DVFS_H

#include <cstdlib>

#include <sstream>

namespace Dvfs
{
   enum CPU_FRQ
   {
      FRQ_102000KHz = 102000,
      FRQ_204000KHz = 204000,
      FRQ_307200KHz = 307200,
      FRQ_403200KHz = 403200,
      FRQ_518400KHz = 518400,
      FRQ_614400KHz = 614400,
      FRQ_710400KHz = 710400,
      FRQ_825600KHz = 825600, 
      FRQ_921600KHz = 921600, 
      FRQ_1036800KHz = 1036800, 
      FRQ_1132800KHz = 1132800, 
      FRQ_1224000KHz = 1224000, 
      FRQ_1326000KHz = 1326000, 
      FRQ_1428000KHz = 1428000, 
      FRQ_1479000KHz = 1479000
   };

   enum GPU_FRQ
   {  
      FRQ_76800000Hz =  76800000,
      FRQ_153600000Hz = 153600000,
      FRQ_230400000Hz = 230400000,
      FRQ_307200000Hz = 307200000,
      FRQ_384000000Hz = 384000000,
      FRQ_460800000Hz = 460800000,
      FRQ_537600000Hz = 537600000,
      FRQ_614400000Hz = 614400000,
      FRQ_691200000Hz = 691200000,
      FRQ_768000000Hz = 768000000,
      FRQ_844800000Hz = 844800000,
      FRQ_921600000Hz = 921600000
   };

   int SetCpuFreq(CPU_FRQ freq){
      std::stringstream command;
      command
         << "sudo echo \"userspace\" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor && echo "
         << freq << " | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed && echo "
         << freq << " | sudo tee /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed && echo "
         << freq << " | sudo tee /sys/devices/system/cpu/cpu2/cpufreq/scaling_setspeed && echo "
         << freq << " | sudo tee /sys/devices/system/cpu/cpu3/cpufreq/scaling_setspeed" 
         << std::endl;
      return std::system(command.str().c_str());
   };
   int SetGpuFreq(GPU_FRQ freq){
      std::stringstream command;
      command
         << "sudo echo \"userspace\" | sudo tee /sys/devices/gpu.0/devfreq/57000000.gpu/governor && echo "
         << FRQ_76800000Hz << " | sudo tee /sys/devices/gpu.0/devfreq/57000000.gpu/min_freq && echo "
         << freq  << " | sudo tee /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq && echo "
         << freq << " | sudo tee /sys/devices/gpu.0/devfreq/57000000.gpu/min_freq" 
         << std::endl;
      return std::system(command.str().c_str());
   };
}

#endif //COMMON_DVFS_DVFS_H