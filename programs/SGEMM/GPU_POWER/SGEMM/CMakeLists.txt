# When the project is built, build this library
add_executable(SGEMM_GpuPower
        "src/sgemm.cc"
)

# Rename the executable file
set_target_properties(SGEMM_GpuPower 
    PROPERTIES 
        OUTPUT_NAME "SGEMM")

# Add sources to the target
#target_sources(SGEMM_Algorithm
#    PRIVATE
#        "src/sgemm.cc"
#)

# Add to the include path of the target
target_include_directories(SGEMM_GpuPower 
    PUBLIC 
        "include"
)
target_link_libraries(SGEMM_GpuPower
    PUBLIC
        SGEMM_Sgemm_LIB
        SGEMM_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(SGEMM_GpuPower PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(SGEMM_GpuPower PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(SGEMM_GpuPower PUBLIC AppRegister_Common_Lib)
target_link_libraries(SGEMM_GpuPower PUBLIC AppRegister_Client_Lib)
