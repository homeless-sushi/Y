# When the project is built, build this library
add_executable(HISTO_GpuPower
    "src/histo.cc"
)

# Rename the executable file
set_target_properties(HISTO_GpuPower 
    PROPERTIES 
        OUTPUT_NAME "HISTO")

# Add sources to the target
#target_sources(HISTO_GpuPower
#    PRIVATE
#        "src/histo.cc"
#)

# Add to the include path of the target
target_include_directories(HISTO_GpuPower 
    PUBLIC 
        "include"
)
target_link_libraries(HISTO_GpuPower
    PUBLIC
        HISTO_Histo_LIB
        HISTO_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(HISTO_GpuPower PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(HISTO_GpuPower PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(HISTO_GpuPower PUBLIC AppRegister_Common_Lib)
target_link_libraries(HISTO_GpuPower PUBLIC AppRegister_Client_Lib)
