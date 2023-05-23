# When the project is built, create this library
add_library(COMMON_Vector_Lib INTERFACE)

# Add to the include path
target_include_directories(COMMON_Vector_Lib 
    INTERFACE
        "include"
)
