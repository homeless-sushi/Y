# When the project is built, build this library
add_executable(SGEMM_Benchmark
        "src/sgemm.cc"
)

# Rename the executable file
set_target_properties(SGEMM_Benchmark 
    PROPERTIES 
        OUTPUT_NAME "SGEMM")

# Add sources to the target
#target_sources(SGEMM_Algorithm
#    PRIVATE
#        "src/sgemm.cc"
#)

# Add to the include path of the target
target_include_directories(SGEMM_Benchmark 
    PUBLIC 
        "include"
)
target_link_libraries(SGEMM_Benchmark
    PUBLIC
        SGEMM_Sgemm_LIB
        SGEMM_Knobs_LIB
)

# Add Margot Library
get_filename_component(MARGOT_JSON "margot/margot.json" ABSOLUTE)
get_filename_component(OPS_JSON "margot/ops.json" ABSOLUTE)
margot_heel_generate_interface("${MARGOT_JSON}")
target_link_libraries(SGEMM_Benchmark PRIVATE margot::margot_heel_interface ${CUDA_LIBRARIES})
target_compile_features(SGEMM_Benchmark PRIVATE cxx_std_17 )

# Add Boost Libraries
target_include_directories(SGEMM_Benchmark PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(SGEMM_Benchmark PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(SGEMM_Benchmark PUBLIC AppRegister_Common_Lib)
target_link_libraries(SGEMM_Benchmark PUBLIC AppRegister_Client_Lib)
