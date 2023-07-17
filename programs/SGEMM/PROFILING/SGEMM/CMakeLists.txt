# When the project is built, build this library
add_executable(SGEMM_Profiling
        "src/sgemm.cc"
)

# Rename the executable file
set_target_properties(SGEMM_Profiling 
    PROPERTIES 
        OUTPUT_NAME "SGEMM")

# Add sources to the target
#target_sources(SGEMM_Algorithm
#    PRIVATE
#        "src/sgemm.cc"
#)

# Add to the include path of the target
target_include_directories(SGEMM_Profiling 
    PUBLIC 
        "include"
)
target_link_libraries(SGEMM_Profiling
    PUBLIC
        SGEMM_Sgemm_LIB
        SGEMM_Knobs_LIB
)

# Add Margot Library
get_filename_component(MARGOT_JSON "margot/margot.json" ABSOLUTE)
margot_heel_generate_interface("${MARGOT_JSON}")
target_link_libraries(SGEMM_Profiling PRIVATE margot::margot_heel_interface ${CUDA_LIBRARIES})
target_compile_features(SGEMM_Profiling PRIVATE cxx_std_17 )

# Add Boost Libraries
target_include_directories(SGEMM_Profiling PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(SGEMM_Profiling PUBLIC ${Boost_LIBRARIES})
