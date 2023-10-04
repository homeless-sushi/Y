# When the project is built, build this library
add_executable(HISTO_Algorithm
    "src/histo.cc"
)

# Rename the executable file
set_target_properties(HISTO_Algorithm 
    PROPERTIES 
        OUTPUT_NAME "HISTO")

# Add sources to the target
#target_sources(HISTO_Algorithm
#    PRIVATE
#        "src/histo.cc"
#)

# Add to the include path of the target
target_include_directories(HISTO_Algorithm 
    PUBLIC 
        "include"
)
target_link_libraries(HISTO_Algorithm
    PUBLIC
        HISTO_Histo_LIB
        HISTO_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(HISTO_Algorithm PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(HISTO_Algorithm PUBLIC ${Boost_LIBRARIES})
