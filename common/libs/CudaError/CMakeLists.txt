# When the project is built, create this library
add_library(COMMON_CudaError_Lib INTERFACE)

# Add to the include path
target_include_directories(COMMON_CudaError_Lib 
    INTERFACE
        "include"
)
