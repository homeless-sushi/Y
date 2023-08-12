# When the project is built, create this library
add_library(SWAPTIONS_SwaptionsCpu_LIB)

# Add executable
target_sources(SWAPTIONS_SwaptionsCpu_LIB
    PRIVATE
        "src/HJM.cc"
        "src/Kernel.cc"
)

# Add to the include path
target_include_directories(SWAPTIONS_SwaptionsCpu_LIB 
    PUBLIC
        "include"
)

# Add library dependency
target_link_libraries(SWAPTIONS_SwaptionsCpu_LIB SWAPTIONS_Swaptions_LIB)
