# When the project is built, build this library
add_executable(CUTCP_Benchmark
    "src/cutcp.cc"
)

# Rename the executable file
set_target_properties(CUTCP_Benchmark
    PROPERTIES 
        OUTPUT_NAME "CUTCP")

# Add sources to the target
#target_sources(CUTCP_Benchmark
#    PRIVATE
#        "src/cutcp.cc"
#)

# Add to the include path of the target
target_include_directories(CUTCP_Benchmark 
    PUBLIC 
        "include"
)
target_link_libraries(CUTCP_Benchmark
    PUBLIC
        CUTCP_Atom_LIB
        CUTCP_Cutcp_LIB
        CUTCP_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(CUTCP_Benchmark PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(CUTCP_Benchmark PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(CUTCP_Benchmark PUBLIC AppRegister_Common_Lib)
target_link_libraries(CUTCP_Benchmark PUBLIC AppRegister_Client_Lib)