# When the project is built, create this library
add_library(CUTCP_Atom_LIB)

# Add executable
target_sources(CUTCP_Atom_LIB
    PRIVATE
        "src/ReadWrite.cc"
        "src/Utils.cc"
)

# Add to the include path
target_include_directories(CUTCP_Atom_LIB 
    PUBLIC
        "include"
)

# Add library dependency
target_link_libraries(CUTCP_Atom_LIB COMMON_Vector_Lib)
