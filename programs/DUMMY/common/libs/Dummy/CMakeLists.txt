# When the project is built, create this library
add_library(DUMMY_Dummy_LIB)

# Add executable
target_sources(DUMMY_Dummy_LIB
    PRIVATE
        "src/Dummy.cu"
        "src/ReadWrite.cc"
)

# Add to the include path
target_include_directories(DUMMY_Dummy_LIB
    PUBLIC
        "include"
)

# Add library dependency
target_link_libraries(DUMMY_Dummy_LIB COMMON_CudaError_Lib)
