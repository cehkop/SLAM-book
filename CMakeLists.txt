# Add header file
cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package (OpenCV 4.0.0 REQUIRED)
# find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# include_directories ("/usr/include/opencv4")
include_directories( "/usr/include/eigen3/Eigen" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/include/opencv4")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ilopencv_highgui")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lopencv_core")

# add_executable( DisplayImage opencv_get_value.cpp )
# target_link_libraries( DisplayImage ${OpenCV_LIBS} )
