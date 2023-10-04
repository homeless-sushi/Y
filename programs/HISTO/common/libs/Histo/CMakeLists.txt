# When the project is built, create this library
add_library(HISTO_Histo_LIB
    "src/Histo.cc"
    "src/HistoCpu.cc"
    "src/HistoCuda.cu"
    "src/ReadWrite.cc"
)

# Add to the include path
target_include_directories(HISTO_Histo_LIB 
    PUBLIC
        "include"
)

# Add library dependency
target_link_libraries(HISTO_Histo_LIB COMMON_CudaError_Lib)
