# When the project is built, build this library
add_executable(HISTO_MargotKnowledge2
    "src/histo.cc"
)

# Rename the executable file
set_target_properties(HISTO_MargotKnowledge2 
    PROPERTIES 
        OUTPUT_NAME "HISTO")

# Add sources to the target
#target_sources(HISTO_MargotKnowledge2
#    PRIVATE
#        "src/histo.cc"
#)

# Add to the include path of the target
target_include_directories(HISTO_MargotKnowledge2 
    PUBLIC 
        "include"
)
target_link_libraries(HISTO_MargotKnowledge2
    PUBLIC
        HISTO_Histo_LIB
        HISTO_Knobs_LIB
)

# Add Margot Library
get_filename_component(MARGOT_JSON "margot/margot.json" ABSOLUTE)
get_filename_component(OPS_JSON "margot/ops.json" ABSOLUTE)
margot_heel_generate_interface("${MARGOT_JSON}" "${OPS_JSON}")
target_link_libraries(HISTO_MargotKnowledge2 PUBLIC margot::margot_heel_interface)
target_compile_features(HISTO_MargotKnowledge2 PRIVATE cxx_std_17)

# Add Boost Libraries
target_include_directories(HISTO_MargotKnowledge2 PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(HISTO_MargotKnowledge2 PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(HISTO_MargotKnowledge2 PUBLIC AppRegister_Server_Lib)
target_link_libraries(HISTO_MargotKnowledge2 PUBLIC AppRegister_Common_Lib)
target_link_libraries(HISTO_MargotKnowledge2 PUBLIC AppRegister_Client_Lib)