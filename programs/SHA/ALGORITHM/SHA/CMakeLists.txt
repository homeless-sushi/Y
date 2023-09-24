# When the project is built, build this library
add_executable(SHA_Algorithm
    "src/sha.cc"
)

# Rename the executable file
set_target_properties(SHA_Algorithm
    PROPERTIES 
        OUTPUT_NAME "SHA")

# Add sources to the target
#target_sources(SHA_Algorithm
#    PRIVATE
#        "src/sha.cc"
#)

# Add to the include path of the target
target_include_directories(SHA_Algorithm 
    PUBLIC 
        "include"
)
target_link_libraries(SHA_Algorithm
    PUBLIC
        SHA_Sha_LIB
)

# Add Boost Libraries
target_include_directories(SHA_Algorithm PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(SHA_Algorithm PUBLIC ${Boost_LIBRARIES})
