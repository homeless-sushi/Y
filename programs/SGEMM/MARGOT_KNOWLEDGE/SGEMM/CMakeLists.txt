# When the project is built, build this library
add_executable(SGEMM_MargotKnowledge
    "src/sgemm.cc"
)

# Rename the executable file
set_target_properties(SGEMM_MargotKnowledge 
    PROPERTIES 
        OUTPUT_NAME "SGEMM")

# Add sources to the target
#target_sources(SGEMM_MargotKnowledge
#    PRIVATE
#        "src/sgemm.cc"
#)

# Add to the include path of the target
target_include_directories(SGEMM_MargotKnowledge 
    PUBLIC 
        "include"
)
target_link_libraries(SGEMM_MargotKnowledge
    PUBLIC
        SGEMM_Sgemm_LIB
        SGEMM_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(SGEMM_MargotKnowledge PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(SGEMM_MargotKnowledge PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(SGEMM_MargotKnowledge PUBLIC AppRegister_Server_Lib)
