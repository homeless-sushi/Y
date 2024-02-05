# When the project is built, build this library
add_executable(NBODY_MargotKnowledge
    "src/nbody.cc"
)

# Rename the executable file
set_target_properties(NBODY_MargotKnowledge 
    PROPERTIES 
        OUTPUT_NAME "NBODY")

# Add sources to the target
#target_sources(NBODY_MargotKnowledge
#    PRIVATE
#        "src/nbody.cc"
#)

# Add to the include path of the target
target_include_directories(NBODY_MargotKnowledge 
    PUBLIC 
        "include"
)
target_link_libraries(NBODY_MargotKnowledge
    PUBLIC
        NBODY_Nbody_LIB
        NBODY_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(NBODY_MargotKnowledge PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(NBODY_MargotKnowledge PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(NBODY_MargotKnowledge PUBLIC AppRegister_Server_Lib)
