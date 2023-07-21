# When the project is built, create this library
add_library(COMMON_Dvfs_Lib INTERFACE)

# Add to the include path
target_include_directories(COMMON_Dvfs_Lib 
    INTERFACE
        "include"
)
