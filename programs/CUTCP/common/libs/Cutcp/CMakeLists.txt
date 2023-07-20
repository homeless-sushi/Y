# When the project is built, create this library
add_library(CUTCP_Cutcp_LIB)

# Add executable
target_sources(CUTCP_Cutcp_LIB
    PRIVATE
        "src/CutcpCpu.cc"
        "src/CutcpCuda.cu"
        "src/ReadWrite.cc"
)

# Add to the include path
target_include_directories(CUTCP_Cutcp_LIB 
    PUBLIC
        "include"
)

# Add Atom dependency
target_link_libraries(CUTCP_Cutcp_LIB COMMON_Vector_Lib)
target_link_libraries(CUTCP_Cutcp_LIB COMMON_CudaError_Lib)
target_link_libraries(CUTCP_Cutcp_LIB CUTCP_Atom_LIB)
