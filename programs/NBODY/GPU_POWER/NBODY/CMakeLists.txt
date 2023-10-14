# When the project is built, build this library
add_executable(NBODY_GpuPower
    "src/nbody.cc"
)

# Rename the executable file
set_target_properties(NBODY_GpuPower 
    PROPERTIES 
        OUTPUT_NAME "NBODY")

# Add sources to the target
#target_sources(NBODY_GpuPower
#    PRIVATE
#        "src/nbody.cc"
#)

# Add to the include path of the target
target_include_directories(NBODY_GpuPower 
    PUBLIC 
        "include"
)
target_link_libraries(NBODY_GpuPower
    PUBLIC
        NBODY_Nbody_LIB
        NBODY_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(NBODY_GpuPower PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(NBODY_GpuPower PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(NBODY_GpuPower PUBLIC AppRegister_Common_Lib)
target_link_libraries(NBODY_GpuPower PUBLIC AppRegister_Client_Lib)
