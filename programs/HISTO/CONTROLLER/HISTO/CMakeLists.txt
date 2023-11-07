# When the project is built, build this library
add_executable(HISTO_Controller
    "src/histo.cc"
)

# Rename the executable file
set_target_properties(HISTO_Controller 
    PROPERTIES 
        OUTPUT_NAME "HISTO")

# Add sources to the target
#target_sources(HISTO_Controller
#    PRIVATE
#        "src/histo.cc"
#)

# Add to the include path of the target
target_include_directories(HISTO_Controller 
    PUBLIC 
        "include"
)
target_link_libraries(HISTO_Controller
    PUBLIC
        HISTO_Histo_LIB
        HISTO_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(HISTO_Controller PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(HISTO_Controller PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(HISTO_Controller PUBLIC AppRegister_Common_Lib)
target_link_libraries(HISTO_Controller PUBLIC AppRegister_Client_Lib)
