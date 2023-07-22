# When the project is built, build this library
add_executable(CUTCP_Profiling
    "src/cutcp.cc"
)

# Rename the executable file
set_target_properties(CUTCP_Profiling
    PROPERTIES 
        OUTPUT_NAME "CUTCP")

# Add sources to the target
#target_sources(CUTCP_Profiling
#    PRIVATE
#        "src/cutcp.cc"
#)

# Add to the include path of the target
target_include_directories(CUTCP_Profiling 
    PUBLIC 
        "include"
)
target_link_libraries(CUTCP_Profiling
    PUBLIC
        CUTCP_Atom_LIB
        CUTCP_Cutcp_LIB
        CUTCP_Knobs_LIB
)

# Add Common Library
target_link_libraries(CUTCP_Profiling PUBLIC COMMON_Dvfs_Lib)

# Add Margot Library
get_filename_component(MARGOT_JSON "margot/margot.json" ABSOLUTE)
margot_heel_generate_interface("${MARGOT_JSON}")
target_link_libraries(CUTCP_Profiling PRIVATE margot::margot_heel_interface ${CUDA_LIBRARIES})
target_compile_features(CUTCP_Profiling PRIVATE cxx_std_17 )

# Add Boost Libraries
target_include_directories(CUTCP_Profiling PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(CUTCP_Profiling PUBLIC ${Boost_LIBRARIES})
