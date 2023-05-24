# When the project is built, build this library
add_executable(NBODY_Profiling
    "src/nbody.cc"
)

# Rename the executable file
set_target_properties(NBODY_Profiling 
    PROPERTIES 
        OUTPUT_NAME "NBODY")

# Add sources to the target
#target_sources(NBODY_Profiling
#    PRIVATE
#        "src/nbody.cc"
#)

# Add to the include path of the target
target_include_directories(NBODY_Profiling 
    PUBLIC 
        "include"
)
target_link_libraries(NBODY_Profiling
    PUBLIC
        NBODY_Nbody_LIB
        NBODY_Knobs_LIB
)

# Add Margot Library
get_filename_component(MARGOT_JSON "margot/margot.json" ABSOLUTE)
margot_heel_generate_interface("${MARGOT_JSON}")
target_link_libraries(NBODY_Profiling PRIVATE margot::margot_heel_interface ${CUDA_LIBRARIES})
target_compile_features(NBODY_Profiling PRIVATE cxx_std_17 )

# Add Boost Libraries
target_include_directories(NBODY_Profiling PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(NBODY_Profiling PUBLIC ${Boost_LIBRARIES})
