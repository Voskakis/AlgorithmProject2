cmake_minimum_required(VERSION 3.16)
project(lsh)
set(CMAKE_CXX_STANDARD 11)

find_package(pybind11 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/myheader.cpp
    ${CMAKE_SOURCE_DIR}/src/bindings.cpp
)

add_library(add_module MODULE ${SOURCES})

target_link_libraries(add_module PRIVATE pybind11::module)

if (WIN32)
    set_target_properties(add_module PROPERTIES PREFIX "" SUFFIX ".pyd")
else()
    set_target_properties(add_module PROPERTIES PREFIX "" SUFFIX ".so")
endif()
