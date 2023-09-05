# When the project is built, create this library
add_library(NBODY_Knobs_LIB
    "src/Device.cc"
    "src/Precision.cc"
)

# Add executable
#target_sources(NBODY_Knobs_LIB
#    PRIVATE
#        "src/Device.cc"
#        "src/Precision.cc"
#)

# Add to the include path
target_include_directories(NBODY_Knobs_LIB 
    PUBLIC
        "include"
)
