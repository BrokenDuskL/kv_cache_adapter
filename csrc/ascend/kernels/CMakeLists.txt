if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
else()
  message(FATAL_ERROR "ascendc_kernel_cmake does not exist; check your CANN installation")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

file(GLOB KERNEL_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

ascendc_library(kv_cache_adapter_npu_custom_kernels SHARED ${KERNEL_FILES})

target_include_directories(kv_cache_adapter_npu_custom_kernels
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
