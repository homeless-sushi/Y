# When the project is built, create this library
add_library(SGEMM_Knobs_LIB)

# Add executable
target_sources(SGEMM_Knobs_LIB
    PRIVATE
        "src/Device.cc"
)

# Add to the include path
target_include_directories(SGEMM_Knobs_LIB 
    PUBLIC
        "include"
)
