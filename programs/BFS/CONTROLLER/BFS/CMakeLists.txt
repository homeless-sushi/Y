# When the project is built, build this library
add_executable(BFS_Controller
    "src/bfs.cc"
)

# Rename the executable file
set_target_properties(BFS_Controller 
    PROPERTIES 
        OUTPUT_NAME "BFS")

# Add sources to the target
#target_sources(BFS_Controller
#    PRIVATE
#        "src/bfs.cc"
#)

# Add to the include path of the target
target_include_directories(BFS_Controller 
    PUBLIC 
        "include"
)
target_link_libraries(BFS_Controller
    PUBLIC
        BFS_Graph_LIB
        BFS_Bfs_LIB
        BFS_Knobs_LIB
)

# Add Boost Libraries
target_include_directories(BFS_Controller PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(BFS_Controller PUBLIC ${Boost_LIBRARIES})

# Add runtime manager dependencies
target_link_libraries(BFS_Controller PUBLIC AppRegister_Common_Lib)
target_link_libraries(BFS_Controller PUBLIC AppRegister_Client_Lib)
