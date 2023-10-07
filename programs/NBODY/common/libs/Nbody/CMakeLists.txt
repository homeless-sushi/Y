# When the project is built, create this library
add_library(NBODY_Nbody_LIB
    "src/NbodyCpu.cc"
    "src/NbodyCuda.cu"
    "src/ReadWrite.cc"
)

# Add executable
#target_sources(NBODY_Nbody_LIB
#    PRIVATE
#        "src/NbodyCpu.cc"
#        "src/NbodyCuda.cu"
#        "src/ReadWrite.cc"
#)

# Add to the include path
target_include_directories(NBODY_Nbody_LIB 
    PUBLIC
        "include"
)

# Add library dependency
target_link_libraries(NBODY_Nbody_LIB COMMON_CudaError_Lib)
target_link_libraries(NBODY_Nbody_LIB COMMON_Vector_Lib)
