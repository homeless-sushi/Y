# When the project is built, build this library
add_executable(NBODY_Algorithm
        "src/nbody.cc"
)

# Rename the executable file
set_target_properties(NBODY_Algorithm 
    PROPERTIES 
        OUTPUT_NAME "NBODY")

# Add sources to the target
#target_sources(NBODY_Algorithm
#    PRIVATE
#        "src/nbody.cc"
#)

# Add to the include path of the target
target_include_directories(NBODY_Algorithm 
    PUBLIC 
        "include"
)
target_link_libraries(NBODY_Algorithm
    PUBLIC
        NBODY_Nbody_LIB
        NBODY_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(NBODY_Algorithm PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(NBODY_Algorithm PUBLIC ${Boost_LIBRARIES})
