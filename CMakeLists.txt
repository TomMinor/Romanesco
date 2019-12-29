cmake_minimum_required(VERSION 2.8.11)

project(romanesco)

# Force C++11
if(UNIX)
  SET(CMAKE_CXX_FLAGS "-std=c++0x")
endif()

add_subdirectory(src)