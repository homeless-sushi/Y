# When the project is built, create this library
add_library(SWAPTIONS_SwaptionsGpu_LIB)

# Add executable
target_sources(SWAPTIONS_SwaptionsGpu_LIB
    PRIVATE
        "src/HJM.cu"
        "src/Kernel.cc"
)

# Add to the include path
target_include_directories(SWAPTIONS_SwaptionsGpu_LIB 
    PUBLIC
        "include"
)

# Add library dependency
target_link_libraries(SWAPTIONS_SwaptionsGpu_LIB SWAPTIONS_Swaptions_LIB)
target_link_libraries(SWAPTIONS_SwaptionsGpu_LIB COMMON_CudaError_Lib)
