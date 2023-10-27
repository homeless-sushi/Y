# When the project is built, build this library
add_executable(SGEMM_Config
        "src/sgemm.cc"
)

# Rename the executable file
set_target_properties(SGEMM_Config 
    PROPERTIES 
        OUTPUT_NAME "SGEMM")

# Add sources to the target
#target_sources(SGEMM_Algorithm
#    PRIVATE
#        "src/sgemm.cc"
#)

# Add to the include path of the target
target_include_directories(SGEMM_Config 
    PUBLIC 
        "include"
)
target_link_libraries(SGEMM_Config
    PUBLIC
        SGEMM_Sgemm_LIB
        SGEMM_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(SGEMM_Config PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(SGEMM_Config PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(SGEMM_Config PUBLIC AppRegister_Common_Lib)
target_link_libraries(SGEMM_Config PUBLIC AppRegister_Client_Lib)
