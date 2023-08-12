# When the project is built, create this library
add_library(SWAPTIONS_Knobs_LIB)

# Add executable
target_sources(SWAPTIONS_Knobs_LIB
    PRIVATE
        "src/Device.cc"
        "src/Precision.cc"
)

# Add to the include path
target_include_directories(SWAPTIONS_Knobs_LIB 
    PUBLIC
        "include"
)
