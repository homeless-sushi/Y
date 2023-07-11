# When the project is built, build this library
add_executable(BFS_Benchmark
    "src/bfs.cc"
)

# Rename the executable file
set_target_properties(BFS_Benchmark 
    PROPERTIES 
        OUTPUT_NAME "BFS")

# Add sources to the target
#target_sources(BFS_Benchmark
#    PRIVATE
#        "src/bfs.cc"
#)

# Add to the include path of the target
target_include_directories(BFS_Benchmark 
    PUBLIC 
        "include"
)
target_link_libraries(BFS_Benchmark
    PUBLIC
        BFS_Graph_LIB
        BFS_Bfs_LIB
        BFS_Knobs_LIB
)

# Add Margot Library
get_filename_component(MARGOT_JSON "margot/margot.json" ABSOLUTE)
get_filename_component(OPS_JSON "margot/ops.json" ABSOLUTE)
margot_heel_generate_interface("${MARGOT_JSON}")
target_link_libraries(BFS_Benchmark PRIVATE margot::margot_heel_interface ${CUDA_LIBRARIES})
target_compile_features(BFS_Benchmark PRIVATE cxx_std_17 )

# Add Boost Libraries
target_include_directories(BFS_Benchmark PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(BFS_Benchmark PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(BFS_Benchmark PUBLIC AppRegister_Common_Lib)
target_link_libraries(BFS_Benchmark PUBLIC AppRegister_Client_Lib)
