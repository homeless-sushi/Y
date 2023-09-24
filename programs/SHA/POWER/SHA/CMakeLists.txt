# When the project is built, build this library
add_executable(SHA_Power
    "src/sha.cc"
)

# Rename the executable file
set_target_properties(SHA_Power
    PROPERTIES 
        OUTPUT_NAME "SHA")

# Add sources to the target
#target_sources(SHA_Power
#    PRIVATE
#        "src/sha.cc"
#)

# Add to the include path of the target
target_include_directories(SHA_Power 
    PUBLIC 
        "include"
)
target_link_libraries(SHA_Power
    PUBLIC
        SHA_Sha_LIB
)

# Add Boost Libraries
target_include_directories(SHA_Power PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(SHA_Power PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(SHA_Power PUBLIC AppRegister_Common_Lib)
target_link_libraries(SHA_Power PUBLIC AppRegister_Client_Lib)
