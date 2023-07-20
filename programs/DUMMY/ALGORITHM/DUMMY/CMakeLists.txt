# When the project is built, build this library
add_executable(DUMMY_Algorithm
    "src/dummy.cc"
)

# Rename the executable file
set_target_properties(DUMMY_Algorithm 
    PROPERTIES 
        OUTPUT_NAME "DUMMY")

# Add sources to the target
#target_sources(DUMMY_Algorithm
#    PRIVATE
#        "src/dummy.cc"
#)

# Add to the include path of the target
target_include_directories(DUMMY_Algorithm 
    PUBLIC 
        "include"
)
target_link_libraries(DUMMY_Algorithm
    PUBLIC
        COMMON_CudaError_Lib
        DUMMY_Dummy_LIB
)

# Add Boost Libraries
target_include_directories(DUMMY_Algorithm PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(DUMMY_Algorithm PUBLIC ${Boost_LIBRARIES})
