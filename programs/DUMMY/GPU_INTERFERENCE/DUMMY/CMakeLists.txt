# When the project is built, build this library
add_executable(DUMMY_GpuInterference
    "src/dummy.cc"
)

# Rename the executable file
set_target_properties(DUMMY_GpuInterference 
    PROPERTIES 
        OUTPUT_NAME "DUMMY")

# Add sources to the target
#target_sources(DUMMY_GpuInterference
#    PRIVATE
#        "src/dummy.cc"
#)

# Add to the include path of the target
target_include_directories(DUMMY_GpuInterference 
    PUBLIC 
        "include"
)
target_link_libraries(DUMMY_GpuInterference
    PUBLIC
        COMMON_CudaError_Lib
        DUMMY_Dummy_LIB
)

# Add Boost Libraries
target_include_directories(DUMMY_GpuInterference PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(DUMMY_GpuInterference PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(DUMMY_GpuInterference PUBLIC AppRegister_Common_Lib)
target_link_libraries(DUMMY_GpuInterference PUBLIC AppRegister_Client_Lib)
