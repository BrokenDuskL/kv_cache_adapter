cmake_minimum_required(VERSION 3.16.0)
project(kv_cache_adapter_npu_custom_ops)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED KVCA_SLOT_META_BITS)
  set(KVCA_SLOT_META_BITS 8)
endif()
add_compile_definitions(KVCA_SLOT_META_BITS=${KVCA_SLOT_META_BITS})

if(NOT GLIBCXX_USE_CXX11_ABI)
  set(GLIBCXX_USE_CXX11_ABI 0)
endif()
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(ARCH_SUBDIR "aarch64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(ARCH_SUBDIR "x86_64-linux")
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

add_subdirectory(kernels)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

file(GLOB SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

pybind11_add_module(kv_cache_adapter_npu_custom_ops ${SRC_FILES})

target_include_directories(kv_cache_adapter_npu_custom_ops
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
  ${pybind11_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_CANN_PACKAGE_PATH}/include
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/experiment/platform
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include/experiment/ascend_hal
  ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/ascendc/include
  ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/experiment/platform
  ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/experiment/ascend_hal
)

target_link_options(kv_cache_adapter_npu_custom_ops PRIVATE
  "-Wl,-rpath,$ORIGIN"
)

target_link_directories(kv_cache_adapter_npu_custom_ops
  PRIVATE
  ${TORCH_PATH}/lib
  ${TORCH_NPU_PATH}/lib
  ${ASCEND_CANN_PACKAGE_PATH}/lib64
  ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/devlib
)

target_link_libraries(kv_cache_adapter_npu_custom_ops
  PUBLIC
  ${TORCH_LIBRARIES}
  libtorch_npu.so
  kv_cache_adapter_npu_custom_kernels
  ascendcl
  platform
  ascend_hal
  tiling_api
)

install(TARGETS
  kv_cache_adapter_npu_custom_ops
  kv_cache_adapter_npu_custom_kernels
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)
