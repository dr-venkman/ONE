if(NOT BUILD_RUNTIME_INFERENCE)
  return()
endif(NOT BUILD_RUNTIME_INFERENCE)

list(APPEND RUNTIME_INFERENCE_SRCS "src/runtime_inference.cc")
include_directories(../../../3rdparty/jsoncpp/json)
add_executable(onert-runtime-inference-app ${RUNTIME_INFERENCE_SRCS})
target_link_libraries(onert-runtime-inference-app nnfw-dev pthread dl jsoncpp)

install(TARGETS onert-runtime-inference-app DESTINATION bin)