# When the project is built, build this library
add_executable(NBODY_Benchmark)

# Rename the executable file
set_target_properties(NBODY_Benchmark 
    PROPERTIES 
        OUTPUT_NAME "NBODY")

# Add sources to the target
target_sources(NBODY_Benchmark
    PRIVATE
        "src/nbody.cc"
)

# Add to the include path of the target
target_include_directories(NBODY_Benchmark 
    PUBLIC 
        "include"
)
target_link_libraries(NBODY_Benchmark
    PUBLIC
        NBODY_Nbody_LIB
        NBODY_Knobs_LIB
)

# Add Margot Library
get_filename_component(MARGOT_JSON "margot/margot.json" ABSOLUTE)
get_filename_component(OPS_JSON "margot/ops.json" ABSOLUTE)
margot_heel_generate_interface("${MARGOT_JSON}" "${OPS_JSON}")
target_link_libraries(NBODY_Benchmark PRIVATE margot::margot_heel_interface ${CUDA_LIBRARIES})
target_compile_features(NBODY_Benchmark PRIVATE cxx_std_17 )

# Add Boost Libraries
target_include_directories(NBODY_Benchmark PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(NBODY_Benchmark PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(NBODY_Benchmark PUBLIC AppRegister_Common_Lib)
target_link_libraries(NBODY_Benchmark PUBLIC AppRegister_Client_Lib)
