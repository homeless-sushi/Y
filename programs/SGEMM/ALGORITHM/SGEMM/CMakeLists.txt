# When the project is built, build this library
add_executable(SGEMM_Algorithm
        "src/sgemm.cc"
)

# Rename the executable file
set_target_properties(SGEMM_Algorithm 
    PROPERTIES 
        OUTPUT_NAME "SGEMM")

# Add sources to the target
#target_sources(SGEMM_Algorithm
#    PRIVATE
#        "src/sgemm.cc"
#)

# Add to the include path of the target
target_include_directories(SGEMM_Algorithm 
    PUBLIC 
        "include"
)
target_link_libraries(SGEMM_Algorithm
    PUBLIC
        SGEMM_Sgemm_LIB
        SGEMM_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(SGEMM_Algorithm PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(SGEMM_Algorithm PUBLIC ${Boost_LIBRARIES})
