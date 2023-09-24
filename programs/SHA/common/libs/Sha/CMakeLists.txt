# When the project is built, create this library
add_library(SHA_Sha_LIB
    "src/Sha.cc"
)

# Add to the include path
target_include_directories(SHA_Sha_LIB 
    PUBLIC
        "include"
)

# Add library dependency
target_link_libraries(SHA_Sha_LIB)
