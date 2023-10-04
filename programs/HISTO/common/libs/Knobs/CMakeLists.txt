# When the project is built, create this library
add_library(HISTO_Knobs_LIB
    "src/Device.cc"
)

# Add to the include path
target_include_directories(HISTO_Knobs_LIB 
    PUBLIC
        "include"
)

# Add library dependency
target_link_libraries(HISTO_Knobs_LIB COMMON_CudaError_Lib)
