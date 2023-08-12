# When the project is built, build this library
add_executable(SWAPTIONS_Algorithm
    "src/swaptions.cc"
)

# Rename the executable file
set_target_properties(SWAPTIONS_Algorithm
    PROPERTIES 
        OUTPUT_NAME "SWAPTIONS")

# Add sources to the target
#target_sources(SWAPTIONS_Algorithm
#    PRIVATE
#        "src/swaptions.cc"
#)

# Add to the include path of the target
target_include_directories(SWAPTIONS_Algorithm 
    PUBLIC 
        "include"
)
target_link_libraries(SWAPTIONS_Algorithm
    PUBLIC
        SWAPTIONS_Knobs_LIB
        SWAPTIONS_Swaptions_LIB
        SWAPTIONS_SwaptionsCpu_LIB
        SWAPTIONS_SwaptionsGpu_LIB
)

# Add Boost Libraries
target_include_directories(SWAPTIONS_Algorithm PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(SWAPTIONS_Algorithm PUBLIC ${Boost_LIBRARIES})
