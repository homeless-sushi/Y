# When the project is built, create this library
add_library(CUTCP_Knobs_LIB)

# Add executable
target_sources(CUTCP_Knobs_LIB
    PRIVATE
        "src/Device.cc"
        "src/Precision.cc"
)

# Add to the include path
target_include_directories(CUTCP_Knobs_LIB 
    PUBLIC
        "include"
)

# Add dependencies
target_link_libraries(CUTCP_Knobs_LIB COMMON_Vector_Lib)
