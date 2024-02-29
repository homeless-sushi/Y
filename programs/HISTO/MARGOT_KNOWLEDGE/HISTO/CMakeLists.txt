# When the project is built, build this library
add_executable(HISTO_MargotKnowledge
    "src/histo.cc"
)

# Rename the executable file
set_target_properties(HISTO_MargotKnowledge 
    PROPERTIES 
        OUTPUT_NAME "HISTO")

# Add sources to the target
#target_sources(HISTO_MargotKnowledge
#    PRIVATE
#        "src/histo.cc"
#)

# Add to the include path of the target
target_include_directories(HISTO_MargotKnowledge 
    PUBLIC 
        "include"
)
target_link_libraries(HISTO_MargotKnowledge
    PUBLIC
        HISTO_Histo_LIB
        HISTO_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(HISTO_MargotKnowledge PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(HISTO_MargotKnowledge PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(HISTO_MargotKnowledge PUBLIC AppRegister_Server_Lib)
