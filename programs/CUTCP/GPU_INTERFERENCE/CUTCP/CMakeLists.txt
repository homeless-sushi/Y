# When the project is built, build this library
add_executable(CUTCP_GpuInterference
    "src/cutcp.cc"
)

# Rename the executable file
set_target_properties(CUTCP_GpuInterference
    PROPERTIES 
        OUTPUT_NAME "CUTCP")

# Add sources to the target
#target_sources(CUTCP_GpuInterference
#    PRIVATE
#        "src/cutcp.cc"
#)

# Add to the include path of the target
target_include_directories(CUTCP_GpuInterference 
    PUBLIC 
        "include"
)
target_link_libraries(CUTCP_GpuInterference
    PUBLIC
        CUTCP_Atom_LIB
        CUTCP_Cutcp_LIB
        CUTCP_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(CUTCP_GpuInterference PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(CUTCP_GpuInterference PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(CUTCP_GpuInterference PUBLIC AppRegister_Common_Lib)
target_link_libraries(CUTCP_GpuInterference PUBLIC AppRegister_Client_Lib)