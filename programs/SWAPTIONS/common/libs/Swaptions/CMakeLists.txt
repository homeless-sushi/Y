# When the project is built, create this library
add_library(SWAPTIONS_Swaptions_LIB)

# Add executable
target_sources(SWAPTIONS_Swaptions_LIB
    PRIVATE
        "src/NumericalRecipes.cc"
        "src/Probability.cc"
        "src/ReadWrite.cc"
)

# Add to the include path
target_include_directories(SWAPTIONS_Swaptions_LIB 
    PUBLIC
        "include"
)
