# When the project is built, build this library
add_executable(NBODY_Margot
    "src/nbody.cc"
)

# Rename the executable file
set_target_properties(NBODY_Margot 
    PROPERTIES 
        OUTPUT_NAME "NBODY")

# Add sources to the target
#target_sources(NBODY_Margot
#    PRIVATE
#        "src/nbody.cc"
#)

# Add to the include path of the target
target_include_directories(NBODY_Margot 
    PUBLIC 
        "include"
)
target_link_libraries(NBODY_Margot
    PUBLIC
        NBODY_Nbody_LIB
        NBODY_Knobs_LIB
)

# Add Margot Library
get_filename_component(MARGOT_JSON "margot/margot.json" ABSOLUTE)
get_filename_component(OPS_JSON "margot/ops.json" ABSOLUTE)
margot_heel_generate_interface("${MARGOT_JSON}" "${OPS_JSON}")
target_link_libraries(NBODY_Margot PUBLIC margot::margot_heel_interface)
target_compile_features(NBODY_Margot PRIVATE cxx_std_17)

# Add Boost Libraries
target_include_directories(NBODY_Margot PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(NBODY_Margot PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(NBODY_Margot PUBLIC AppRegister_Server_Lib)
target_link_libraries(NBODY_Margot PUBLIC AppRegister_Common_Lib)
target_link_libraries(NBODY_Margot PUBLIC AppRegister_Client_Lib)