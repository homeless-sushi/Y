# When the project is built, build this library
add_executable(CUTCP_Profiling
    "src/cutcp.cc"
)

# Rename the executable file
set_target_properties(CUTCP_Profiling
    PROPERTIES 
        OUTPUT_NAME "CUTCP")

# Add sources to the target
#target_sources(CUTCP_Profiling
#    PRIVATE
#        "src/cutcp.cc"
#)

# Add to the include path of the target
target_include_directories(CUTCP_Profiling 
    PUBLIC 
        "include"
)
target_link_libraries(CUTCP_Profiling
    PUBLIC
        CUTCP_Atom_LIB
        CUTCP_Cutcp_LIB
        CUTCP_Knobs_LIB
)

# Add Common Library
target_link_libraries(CUTCP_Profiling PUBLIC COMMON_Dvfs_Lib)

# Add Boost Libraries
target_include_directories(CUTCP_Profiling PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(CUTCP_Profiling PUBLIC ${Boost_LIBRARIES})
