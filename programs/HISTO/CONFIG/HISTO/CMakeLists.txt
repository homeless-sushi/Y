# When the project is built, build this library
add_executable(HISTO_Config
    "src/histo.cc"
)

# Rename the executable file
set_target_properties(HISTO_Config 
    PROPERTIES 
        OUTPUT_NAME "HISTO")

# Add sources to the target
#target_sources(HISTO_Config
#    PRIVATE
#        "src/histo.cc"
#)

# Add to the include path of the target
target_include_directories(HISTO_Config 
    PUBLIC 
        "include"
)
target_link_libraries(HISTO_Config
    PUBLIC
        HISTO_Histo_LIB
        HISTO_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(HISTO_Config PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(HISTO_Config PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(HISTO_Config PUBLIC AppRegister_Common_Lib)
target_link_libraries(HISTO_Config PUBLIC AppRegister_Client_Lib)
