# When the project is built, build this library
add_executable(CUTCP_Benchmark
    "src/cutcp.cc"
)

# Rename the executable file
set_target_properties(CUTCP_Benchmark
    PROPERTIES 
        OUTPUT_NAME "CUTCP")

# Add sources to the target
#target_sources(CUTCP_Benchmark
#    PRIVATE
#        "src/cutcp.cc"
#)

# Add to the include path of the target
target_include_directories(CUTCP_Benchmark 
    PUBLIC 
        "include"
)
target_link_libraries(CUTCP_Benchmark
    PUBLIC
        CUTCP_Atom_LIB
        CUTCP_Cutcp_LIB
        CUTCP_Knobs_LIB
)

# Add Margot Library
get_filename_component(MARGOT_JSON "margot/margot.json" ABSOLUTE)
get_filename_component(OPS_JSON "margot/ops.json" ABSOLUTE)
margot_heel_generate_interface("${MARGOT_JSON}")
target_link_libraries(CUTCP_Benchmark PRIVATE margot::margot_heel_interface ${CUDA_LIBRARIES})
target_compile_features(CUTCP_Benchmark PRIVATE cxx_std_17 )

# Add Boost Libraries
target_include_directories(CUTCP_Benchmark PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(CUTCP_Benchmark PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(CUTCP_Benchmark PUBLIC AppRegister_Common_Lib)
target_link_libraries(CUTCP_Benchmark PUBLIC AppRegister_Client_Lib)