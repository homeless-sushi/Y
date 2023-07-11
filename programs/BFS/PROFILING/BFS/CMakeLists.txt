# When the project is built, build this library
add_executable(BFS_Profiling
    "src/bfs.cc"
)

# Rename the executable file
set_target_properties(BFS_Profiling 
    PROPERTIES 
        OUTPUT_NAME "BFS")

# Add sources to the target
#target_sources(BFS_Profiling
#    PRIVATE
#        "src/bfs.cc"
#)

# Add to the include path of the target
target_include_directories(BFS_Profiling 
    PUBLIC 
        "include"
)
target_link_libraries(BFS_Profiling
    PUBLIC
        BFS_Graph_LIB
        BFS_Bfs_LIB
        BFS_Knobs_LIB
)

# Add Margot Library
get_filename_component(MARGOT_JSON "margot/margot.json" ABSOLUTE)
margot_heel_generate_interface("${MARGOT_JSON}")
target_link_libraries(BFS_Profiling PRIVATE margot::margot_heel_interface ${CUDA_LIBRARIES})
target_compile_features(BFS_Profiling PRIVATE cxx_std_17 )

# Add Boost Libraries
target_include_directories(BFS_Profiling PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(BFS_Profiling PUBLIC ${Boost_LIBRARIES})
