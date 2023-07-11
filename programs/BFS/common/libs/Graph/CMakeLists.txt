# When the project is built, create this library
add_library(BFS_Graph_LIB)

# Add sources
target_sources(BFS_Graph_LIB
    PRIVATE
        "src/Graph.cc"
        "src/ReadWrite.cc"
)

# Add to the include path
target_include_directories(BFS_Graph_LIB 
    PUBLIC
        "include"
)
