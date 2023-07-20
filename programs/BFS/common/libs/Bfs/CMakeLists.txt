# When the project is built, create this library
add_library(BFS_Bfs_LIB)

# Add sources
target_sources(BFS_Bfs_LIB
    PRIVATE
        "src/Bfs.cc"
        "src/BfsCpu.cc"
        "src/BfsCuda.cu"
        "src/ReadWrite.cc"
)

# Add to the include path
target_include_directories(BFS_Bfs_LIB 
    PUBLIC
        "include"
)

# Add library dependency
target_link_libraries(BFS_Bfs_LIB COMMON_CudaError_Lib)
target_link_libraries(BFS_Bfs_LIB BFS_Graph_LIB)
