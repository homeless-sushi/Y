# When the project is built, build this library
add_executable(NBODY_GpuInterference
    "src/nbody.cc"
)

# Rename the executable file
set_target_properties(NBODY_GpuInterference 
    PROPERTIES 
        OUTPUT_NAME "NBODY")

# Add sources to the target
#target_sources(NBODY_GpuInterference
#    PRIVATE
#        "src/nbody.cc"
#)

# Add to the include path of the target
target_include_directories(NBODY_GpuInterference 
    PUBLIC 
        "include"
)
target_link_libraries(NBODY_GpuInterference
    PUBLIC
        NBODY_Nbody_LIB
        NBODY_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(NBODY_GpuInterference PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(NBODY_GpuInterference PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(NBODY_GpuInterference PUBLIC AppRegister_Common_Lib)
target_link_libraries(NBODY_GpuInterference PUBLIC AppRegister_Client_Lib)
