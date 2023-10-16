# When the project is built, build this library
add_executable(CUTCP_Controller
    "src/cutcp.cc"
)

# Rename the executable file
set_target_properties(CUTCP_Controller
    PROPERTIES 
        OUTPUT_NAME "CUTCP")

# Add sources to the target
#target_sources(CUTCP_Controller
#    PRIVATE
#        "src/cutcp.cc"
#)

# Add to the include path of the target
target_include_directories(CUTCP_Controller 
    PUBLIC 
        "include"
)
target_link_libraries(CUTCP_Controller
    PUBLIC
        CUTCP_Atom_LIB
        CUTCP_Cutcp_LIB
        CUTCP_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(CUTCP_Controller PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(CUTCP_Controller PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(CUTCP_Controller PUBLIC AppRegister_Common_Lib)
target_link_libraries(CUTCP_Controller PUBLIC AppRegister_Client_Lib)