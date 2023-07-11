# When the project is built, build this library
add_executable(BFS_Algorithm
    "src/bfs.cc"
)

# Rename the executable file
set_target_properties(BFS_Algorithm 
    PROPERTIES 
        OUTPUT_NAME "BFS")

# Add sources to the target
#target_sources(BFS_Algorithm
#    PRIVATE
#        "src/bfs.cc"
#)

# Add to the include path of the target
target_include_directories(BFS_Algorithm 
    PUBLIC 
        "include"
)
target_link_libraries(BFS_Algorithm
    PUBLIC
        BFS_Graph_LIB
        BFS_Bfs_LIB
        BFS_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(BFS_Algorithm PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(BFS_Algorithm PUBLIC ${Boost_LIBRARIES})
