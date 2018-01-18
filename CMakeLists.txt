cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(cjieba)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

set(CMAKE_MACOSX_RPATH 1)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(fasttext EXCLUDE_FROM_ALL)

include_directories(
    ${PROJECT_SOURCE_DIR}/fasttext/src
    ${PROJECT_SOURCE_DIR}/include)


set(SOURCES
    ${PROJECT_SOURCE_DIR}/lib/cfasttext.cc)

add_library(objlib OBJECT ${SOURCES})

add_library(cfasttext SHARED $<TARGET_OBJECTS:objlib>)
add_library(cfasttext_static STATIC $<TARGET_OBJECTS:objlib>)

set_target_properties(cfasttext PROPERTIES PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/include/cfasttext.h)

target_link_libraries(cfasttext fasttext-static_pic)
target_link_libraries(cfasttext_static fasttext-static_pic)

