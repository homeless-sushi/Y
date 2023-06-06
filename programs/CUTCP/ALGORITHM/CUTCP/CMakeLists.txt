# When the project is built, build this library
add_executable(CUTCP_Algorithm
    "src/cutcp.cc"
)

# Rename the executable file
set_target_properties(CUTCP_Algorithm
    PROPERTIES 
        OUTPUT_NAME "CUTCP")

# Add sources to the target
#target_sources(CUTCP_Algorithm
#    PRIVATE
#        "src/cutcp.cc"
#)

# Add to the include path of the target
target_include_directories(CUTCP_Algorithm 
    PUBLIC 
        "include"
)
target_link_libraries(CUTCP_Algorithm
    PUBLIC
        CUTCP_Atom_LIB
        CUTCP_Cutcp_LIB
        CUTCP_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(CUTCP_Algorithm PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(CUTCP_Algorithm PUBLIC ${Boost_LIBRARIES})
