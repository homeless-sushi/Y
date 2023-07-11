# When the project is built, create this library
add_library(BFS_Knobs_LIB)

# Add sources
target_sources(BFS_Knobs_LIB
    PRIVATE
        "src/Device.cc"
)

# Add to the include path
target_include_directories(BFS_Knobs_LIB 
    PUBLIC
        "include"
)
