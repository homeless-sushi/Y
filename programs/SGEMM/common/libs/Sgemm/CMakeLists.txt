# When the project is built, create this library
add_library(SGEMM_Sgemm_LIB)

# Add executable
target_sources(SGEMM_Sgemm_LIB
    PRIVATE
        "src/Matrix.cc"
        "src/Sgemm.cc"
        "src/SgemmCpu.cc"
        "src/SgemmCuda.cu"
        "src/ReadWrite.cc"
)

# Add to the include path
target_include_directories(SGEMM_Sgemm_LIB 
    PUBLIC
        "include"
)

# Add library dependency
target_link_libraries(SGEMM_Sgemm_LIB COMMON_CudaError_Lib)
