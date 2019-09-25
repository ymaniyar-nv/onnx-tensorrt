 # Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a
 # copy of this software and associated documentation files (the "Software"),
 # to deal in the Software without restriction, including without limitation
 # the rights to use, copy, modify, merge, publish, distribute, sublicense,
 # and/or sell copies of the Software, and to permit persons to whom the
 # Software is furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 # DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
# The version of CMake which is not compatible with the old CUDA CMake commands.
set(CMAKE_VERSION_THRESHOLD "3.10.0")

include(FindPackageHandleStandardArgs)

#  ToDo Factor this out to a common Cmake Include file.
# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if(NOT DEFINED ${var})
        set(${var} ${val})
    endif()
endmacro()


# Write the libs out to the top-level of the build directory.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${TENSORRT_BUILD})
message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

# Write the libs out to the top-level of the build directory.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${TENSORRT_BUILD})
message(STATUS "CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")


if(${CMAKE_VERSION} VERSION_LESS ${CMAKE_VERSION_THRESHOLD})
  project(onnx2trt LANGUAGES CXX C)
else()
  project(onnx2trt LANGUAGES CXX C CUDA)
endif()

#
# CUDA Configuration
# This is no longer necessary, https://cmake.org/cmake/help/latest/module/FindCUDA.html
# so we will do it only for older versions of cmake
if(${CMAKE_VERSION} VERSION_LESS ${CMAKE_VERSION_THRESHOLD})
  find_package(CUDA REQUIRED)
endif()

if ( MSVC )
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${W10_LIBRARY_SUFFIXES})
endif()

# Allows us to create standalone wheels. Strip debug symbols from release builds:
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_SHARED_LINKER_FLAGS -Wl,-rpath=$ORIGIN)
else()
  set(CMAKE_SHARED_LINKER_FLAGS -Wl,-rpath=$ORIGIN,--strip-all)
endif()

set(ONNX2TRT_ROOT ${PROJECT_SOURCE_DIR})

# Find length of source directory used to pad filename in Status.hpp
string(LENGTH "${CMAKE_SOURCE_DIR}/" SOURCE_LENGTH)
add_definitions("-DSOURCE_LENGTH=${SOURCE_LENGTH}")

# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD  11)
add_compile_options(-std=c++11)
# Enable compiler warnings
if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated-declarations -Wno-unused-function")
endif()
# Enable compiler warnings and build library flag on Windows.
if ( MSVC )
     add_compile_options(-DTENSORRT_BUILD_LIB)
     set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W4 /FS")
endif()

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PARSER_LINKER_SCRIPT  ${ONNX2TRT_ROOT}/libnvonnxparser.version)

#--------------------------------------------------
# Version information
#--------------------------------------------------
set_ifndef(TRT_MAJOR 0)
set_ifndef(TRT_MINOR 1)
set_ifndef(TRT_PATCH 0)

set(ONNX2TRT_MAJOR ${TRT_MAJOR})
set(ONNX2TRT_MINOR ${TRT_MINOR})
set(ONNX2TRT_PATCH ${TRT_PATCH})

#--------------------------------------------------
# Build configurations, global to all projects
#--------------------------------------------------

set(IMPORTER_SOURCES
  NvOnnxParser.cpp
  ModelImporter.cpp
  builtin_op_importers.cpp
  onnx2trt_utils.cpp
  ShapedWeights.cpp
  OnnxAttrs.cpp
)

set(ONNXIFI_SOURCES onnx_trt_backend.cpp)

set(EXECUTABLE_SOURCES
  main.cpp
)

set(HEADERS
  NvOnnxParser.h
)


#FIND_PACKAGE(Protobuf REQUIRED)
#For newer cmakes this will add -pthread to nvcc, we do not need it.
#set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
#set(THREADS_PREFER_PTHREAD_FLAG TRUE)

if (NOT MSVC)
  find_package(Threads REQUIRED)
endif()

#...PROTOBUF_LIBRARY_LOC1 is the location of the executable. Since we do crosscompile this should point to the
#...host machine protoc executable only.
set(PROTOBUF_LIBRARY_LOC1A ${PROTOBUF_SRC_ROOT_FOLDER}/${CMAKE_HOST_SYSTEM_PROCESSOR}/${PROTOBUF_VER})
set(PROTOBUF_LIBRARY_LOC1B ${PROTOBUF_SRC_ROOT_FOLDER}/${PROTOBUF_VER}/${CMAKE_HOST_SYSTEM_PROCESSOR})  # DVS/GVS

#...PROTOBUF_LIBRARY_LOC2 is the location of the includes and the libs, should point to the target machine
set(PROTOBUF_LIBRARY_LOC2 ${PROTOBUF_SRC_ROOT_FOLDER}/$ENV{PROTOBUF_VERSION})

find_program(PROTOC_EXEC NAMES protoc
  HINTS ${PROTOBUF_LIBRARY_LOC1A} ${PROTOBUF_LIBRARY_LOC1B} ${PROTOC_EXEC_LOC}
  PATH_SUFFIXES bin
  [NO_DEFAULT_PATH]
  [NO_CMAKE_ENVIRONMENT_PATH]
  [NO_CMAKE_PATH]
  [NO_SYSTEM_ENVIRONMENT_PATH]
  [NO_CMAKE_SYSTEM_PATH]
  )

message(STATUS "Found PROTOC_EXEC at ${PROTOC_EXEC}")

if ( MSVC )
  find_library(PROTOBUF_LIBRARY NAMES libprotobuf.lib libprotobuf libprotobufd.lib
    HINTS ${PROTOBUF_LIBRARY_LOC}
    PATH_SUFFIXES lib
    [NO_DEFAULT_PATH]
    [NO_CMAKE_ENVIRONMENT_PATH]
    [NO_CMAKE_PATH]
    [NO_SYSTEM_ENVIRONMENT_PATH]
    [NO_CMAKE_SYSTEM_PATH]
    )
else()
  find_library(PROTOBUF_LIBRARY libprotobuf.a
    HINTS ${PROTOBUF_LIBRARY_LOC2}
    PATH_SUFFIXES lib
    [NO_DEFAULT_PATH]
    [NO_CMAKE_ENVIRONMENT_PATH]
    [NO_CMAKE_PATH]
    [NO_SYSTEM_ENVIRONMENT_PATH]
    [NO_CMAKE_SYSTEM_PATH]
    )
endif()

message(STATUS "Found PROTOBUF_LIBRARY at ${PROTOBUF_LIBRARY}")

if(NOT TARGET onnx_proto)
  # Note: This avoids libprotobuf.so complaining about name collisions at runtime
  if(NOT ONNX_NAMESPACE)
    set(ONNX_NAMESPACE "onnx2trt_onnx")
  endif()
  add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
  add_subdirectory(third_party/onnx EXCLUDE_FROM_ALL)
endif()

if(GOOGLE)
  add_definitions("-Dgoogle=${GOOGLE}")
  add_definitions("-DGOOGLE_PROTOBUF_ARCH_64_BIT")
endif()

if(DEFINED TRT_CXX_ABI)
  add_definitions("-D_GLIBCXX_USE_CXX11_ABI=${TRT_CXX_ABI}")
endif()

add_definitions(${ADDITIONAL_PLATFORM_INCL_FLAGS})

#
# CUDA Configuration
#

string(TOLOWER ${CMAKE_SYSTEM_NAME} cmake_system_name)

if(DEFINED W10_CUDA_ROOT)
  set(CUDA_DIR ${W10_CUDA_ROOT})
endif()

if(NOT DEFINED CUDA_DIR)
  set(CUDA_DIR ${CUDA_ROOT})
endif()
message(STATUS "Setting CUDA_DIR to ${CUDA_DIR}")

if ( MSVC )
  STRING(REGEX REPLACE "\\\\" "/" CUDA_DIR ${CUDA_DIR})
  find_library(CUDA_CUDART_LIBRARY NAMES cudart cudart.lib
    HINTS ${CUDA_DIR}
    PATH_SUFFIXES lib64 lib x64
    )
  find_library(CUDA_CUBLAS_LIBRARY NAMES cublas cublas.lib
    HINTS ${CUDA_DIR}
    PATH_SUFFIXES lib64 lib x64
    )
else()
  find_library(CUDA_CUDART_LIBRARY cudart
    HINTS ${CUDA_DIR}
    PATH_SUFFIXES lib64 lib
    )
  find_library(CUDA_CUBLAS_LIBRARY cublas
    HINTS ${CUDA_DIR}
    PATH_SUFFIXES lib64 lib
    )
endif()


message(STATUS "Found cudart at ${CUDA_CUDART_LIBRARY}")
message(STATUS "Found CUDA_ROOT at ${CUDA_ROOT}")
message(STATUS "Found cublas at ${CUDA_CUBLAS_LIBRARY}")


#...This is no longer necessary, https://cmake.org/cmake/help/latest/module/FindCUDA.html
if(${CMAKE_VERSION} VERSION_LESS ${CMAKE_VERSION_THRESHOLD})
  find_package(CUDA REQUIRED)
endif()

#...This is really a hack, find_package CUDA resets CUDA_64_BIT_DEVICE_CODE to OFF
set(CUDA_64_BIT_DEVICE_CODE ON)

message(STATUS "TARGET_ARCHITECTURE =  ${TARGET_ARCHITECTURE}")
message(STATUS "NV_TARGET_OS =  ${NV_TARGET_OS}")

if(DEFINED GPU_ARCHS)
  #...Replace potentially duplicate white spaces with a single " "
  string(REGEX REPLACE "(\ +)" " " GPU_ARCHS ${GPU_ARCHS})

  #...Replace space separated string into a list
  string(REPLACE " " ";" GPU_ARCHS ${GPU_ARCHS})

  list(LENGTH GPU_ARCHS NUMBER_OF_GPU_ARCHS)
  message(STATUS "GPU_ARCHS DEFINED ${GPU_ARCHS}, there are ${NUMBER_OF_GPU_ARCHS} archs in the list")
else()
  message(STATUS "GPU_ARCHS not DEFINED")
endif()
#...if GPU_ARCHS are defined by the makefile, we do not mock with it.
if(NOT DEFINED GPU_ARCHS OR ${NUMBER_OF_GPU_ARCHS} EQUAL 0)
  list(APPEND GPU_ARCHS
    35
    53
    61
    70
    75
    )
  message(STATUS "GPU_ARCHS were not defined, defining the default list GPU_ARCHS = ${GPU_ARCHS}")
endif()

set(CUDA_VERBOSE_BUILD ON)

# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
# Generate PTX for the last architecture
list(GET GPU_ARCHS -1 LATEST_GPU_ARCH)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_GPU_ARCH},code=compute_${LATEST_GPU_ARCH}")

if ( MSVC )
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
     -cudart static \
     -lineinfo \
     --expt-extended-lambda \
     ${GENCODES} \
     ${ADDITIONAL_PLATFORM_INCL_FLAGS} \
  ")

  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        -D_DEBUG \
    ")
  endif()

else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
     -cudart static \
     -lineinfo \
    -g \
     --expt-extended-lambda \
     ${GENCODES} \
   ")

endif()
# Specify the cuda host compiler to use the same compiler as cmake.
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# CUDNN
#.....
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDNN_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda-${CUDA_VERSION}/include include)
find_library(CUDNN_LIBRARY NAMES cudnn cudnn.lib
    HINTS ${CUDNN_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

message(STATUS "Found CUDNN Include Directory at ${CUDNN_INCLUDE_DIR}")
message(STATUS "Found CUDNN Library at ${CUDNN_LIBRARY}")

find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIR CUDNN_LIBRARY)

if(NOT CUDNN_FOUND)
  message(WARNING
      "Cudnn cannot be found. TensorRT depends explicitly "
      "on cudnn so you should consider installing it.")
  return()
endif()

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

message(STATUS "TENSORRT_BUILD = ${TENSORRT_BUILD}")
message(STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")

message(STATUS "CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")

# TENSORRT_LIBRARY_INFER is defined outside with proper path and debug / release suffixes
find_library(TENSORRT_LIBRARY_INFER NAMES nvinfer nvinfer.dll
  HINTS ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64
  NO_DEFAULT_PATH
  NO_CMAKE_PATH)

message(STATUS "TENSORRT_LIBRARY_INFER = ${TENSORRT_LIBRARY_INFER}")

# TENSORRT_LIBRARY_INFER_PLUGIN is defined outside with proper path and debug / release suffixes
find_library(TENSORRT_LIBRARY_INFER_PLUGIN NAMES nvinfer_plugin nvinfer_plugin.dll
  HINTS ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN})
message(STATUS "Found TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()


if ( MSVC )
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd /FS")
# set postfix for non-win10 builds.
else()
  set(CMAKE_DEBUG_POSTFIX _debug)
endif()

# --------------------------------
# Setting var for windows build
# --------------------------------

if(${NV_TARGET_OS} MATCHES "wddm2")
    if(DEFINED W10_LINKER)
      set(CMAKE_LINKER ${W10_LINKER})
    endif()
endif()



# --------------------------------
# Plugin library
# --------------------------------
if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
endif()
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC --expt-extended-lambda")


# this is a hack, cudadevrt exists in CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES linux_x86_64 build
# but for android, CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES is empty
# tested on nvcc V10.0.95, and clang 5.0.300080 in ndk 16b, cmake 3.12.0
if(CMAKE_C_COMPILER_TARGET STREQUAL aarch64-none-linux-android)
    list(APPEND CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES ${CUDA_ROOT}/lib64)
    list(APPEND CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES ${CUDA_ROOT}/targets/aarch64-linux-androideabi/lib)
    list(APPEND CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES cudadevrt)
endif()

if(${CMAKE_VERSION} VERSION_LESS ${CMAKE_VERSION_THRESHOLD})
    CUDA_INCLUDE_DIRECTORIES(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
else()
    include_directories(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
    if(${NV_TARGET_OS} MATCHES "wddm2")
        #...LG: for windows it does not make sense to build static libs
        set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CUDA_TOOLKIT_ROOT_DIR}/include)

    else()
        include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDA_INCLUDE_DIRS} ${CUDA_INC})

    endif()
endif()


# Use both IMPORTER and RUNTIME sources to build only one .dll
if (MSVC)

add_library(nvonnxparser SHARED ${IMPORTER_SOURCES})
target_include_directories(nvonnxparser PUBLIC ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${MSVC_COMPILER_DIR}/include)
target_link_libraries(nvonnxparser PUBLIC onnx_proto ${PROTOBUF_LIBRARY} ${CUDNN_LIBRARY} ${TENSORRT_LIBRARY} ${ADDITIONAL_PLATFORM_LIB_FLAGS})
set_target_properties(nvonnxparser PROPERTIES
  VERSION   ${ONNX2TRT_MAJOR}.${ONNX2TRT_MINOR}.${ONNX2TRT_PATCH}
  SOVERSION ${ONNX2TRT_MAJOR}
  LINK_DEPENDS ${PARSER_LINKER_SCRIPT}
  LINK_FLAGS "-Wl,--version-script=${PARSER_LINKER_SCRIPT}"
)

endif()

if (NOT MSVC )
# --------------------------------
# Importer library
# --------------------------------
add_library(nvonnxparser SHARED ${IMPORTER_SOURCES})
target_include_directories(nvonnxparser PUBLIC ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${MSVC_COMPILER_DIR}/include)
target_link_libraries(nvonnxparser PUBLIC onnx_proto ${PROTOBUF_LIBRARY} ${CUDNN_LIBRARY} ${TENSORRT_LIBRARY})
set_target_properties(nvonnxparser PROPERTIES
  VERSION   ${ONNX2TRT_MAJOR}.${ONNX2TRT_MINOR}.${ONNX2TRT_PATCH}
  SOVERSION ${ONNX2TRT_MAJOR}
  LINK_DEPENDS ${PARSER_LINKER_SCRIPT}
  LINK_FLAGS "-Wl,--version-script=${PARSER_LINKER_SCRIPT}"
)

add_library(nvonnxparser_static STATIC ${IMPORTER_SOURCES})
target_include_directories(nvonnxparser_static PUBLIC ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_link_libraries(nvonnxparser_static PUBLIC onnx_proto ${PROTOBUF_LIBRARY} ${CUDNN_LIBRARY} ${TENSORRT_LIBRARY})

endif()

# # # Disable to ensure compilation for ERIS, etc. Will enable one by one

# # # --------------------------------
# # # Onnxifi library
# # # --------------------------------

# # add_library(trt_onnxify SHARED ${ONNXIFI_SOURCES})
# # target_include_directories(trt_onnxify PUBLIC ${CUDA_INCLUDE_DIRS} ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
# # target_link_libraries(trt_onnxify PUBLIC nvonnxparser_static ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${ADDITIONAL_PLATFORM_LIB_FLAGS})

# # # --------------------------------
# # # Converter executable
# # # --------------------------------
# # add_executable(onnx2trt ${EXECUTABLE_SOURCES})
# # target_include_directories(onnx2trt PUBLIC ${ONNX_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
# # target_link_libraries(onnx2trt PUBLIC nvonnxparser_static ${CUDA_LIBRARIES} ${ADDITIONAL_PLATFORM_LIB_FLAGS})

# # # --------------------------------
# # # Build Python Bindings
# # # --------------------------------
# # find_program(PYTHON "python")
# # find_program(SWIG "swig")


# # #   Disabling SWIG for all since thr ERIS/DVS do not ghave SWIG. this can be done enablibng swig only for DVS or user build
# # #! We can build swig only for platforms for which both python and swig are available.
# # #! in addition, the swig generation is not smart enough to detect a cross platform compilation at this point so we
# # #! explicitly disable it right now.
# # if (PYTHON AND SWIG AND NOT DISABLE_SWIG)
# #   set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
# #   set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
# #   set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/module/__init__.py")
# #   set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

# #   configure_file(${SETUP_PY_IN} ${SETUP_PY})

# #   add_custom_command(OUTPUT ${OUTPUT}
# #     #COMMAND cd .. && ${PYTHON} ${SETUP_PY_IN} build_ext -I ${TENSORRT_INCLUDE_DIR}
# #     COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && pwd && ${PYTHON} ${SETUP_PY_IN} build_ext --build-lib ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} --include-dirs ${TENSORRT_INCLUDE_DIR} #--source-dir ${CMAKE_CURRENT_SOURCE_DIR}
# #     )

# #   add_custom_target(target ALL DEPENDS ${OUTPUT} nvonnxparser)

# #   #    install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
# # endif()

# # --------------------------------
# # Installation
# # --------------------------------
# #install(TARGETS onnx2trt
#                nvonnxparser_static
if (NOT MSVC )
  install(TARGETS nvonnxparser
    nvonnxparser_static
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )
else()
  install(TARGETS nvonnxparser
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
endif()

install(FILES ${HEADERS}
  DESTINATION include
  )


SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Mike Houston") #required
SET(CPACK_PACKAGE_NAME "onnx-trt-dev")
SET(CPACK_PACKAGE_VERSION "0.5.9")
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "5")
SET(CPACK_PACKAGE_VERSION_PATCH "9")

INCLUDE(CPack)
