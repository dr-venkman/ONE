if(${DOWNLOAD_GTEST})
  nnas_find_package(GTestSource QUIET)

  if(NOT GTestSource_FOUND)
    set(GTest_FOUND FALSE)
    return()
  endif(NOT GTestSource_FOUND)

  if(NOT TARGET gtest_main)
    nnas_include(ExternalProjectTools)
    add_extdirectory(${GTestSource_DIR} gtest EXCLUDE_FROM_ALL)
  endif(NOT TARGET gtest_main)

  set(GTest_FOUND TRUE)
  return()
endif(${DOWNLOAD_GTEST})

### Find and use pre-installed Google Test
find_package(GTest)
find_package(Threads)

if(${GTEST_FOUND} AND TARGET Threads::Threads)
  if(NOT TARGET gtest)
    add_library(gtest INTERFACE)
    target_include_directories(gtest INTERFACE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(gtest INTERFACE ${GTEST_LIBRARIES} Threads::Threads)
  endif(NOT TARGET gtest)

  if(NOT TARGET gtest_main)
    add_library(gtest_main INTERFACE)
    target_include_directories(gtest_main INTERFACE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(gtest_main INTERFACE gtest)
    target_link_libraries(gtest_main INTERFACE ${GTEST_MAIN_LIBRARIES})
  endif(NOT TARGET gtest_main)

  if(NOT TARGET gmock)
    find_library(GMOCK_LIBRARIES gmock)
    find_path(GMOCK_INCLUDE_DIR gmock/gmock.h)
    if(GMOCK_LIBRARIES AND GMOCK_INCLUDE_DIR)
      add_library(gmock INTERFACE)
      target_include_directories(gmock INTERFACE ${GMOCK_INCLUDE_DIR})
      target_link_libraries(gmock INTERFACE ${GMOCK_LIBRARIES} Threads::Threads)
    endif(GMOCK_LIBRARIES)
  endif(NOT TARGET gmock)

  if(NOT TARGET gmock_main)
    find_library(GMOCK_MAIN_LIBRARIES gmock_main)
    find_path(GMOCK_INCLUDE_DIR gmock/gmock.h)
    if(GMOCK_MAIN_LIBRARIES AND GMOCK_INCLUDE_DIR)
      add_library(gmock_main INTERFACE)
      target_include_directories(gmock_main INTERFACE ${GMOCK_INCLUDE_DIR})
      target_link_libraries(gmock_main INTERFACE gmock)
      target_link_libraries(gmock_main INTERFACE ${GMOCK_MAIN_LIBRARIES})
    endif(GMOCK_MAIN_LIBRARIES AND GMOCK_INCLUDE_DIR)
  endif(NOT TARGET gmock_main)

  # TODO Check whether this command is necessary or not
  include_directories(${GTEST_INCLUDE_DIR})
  set(GTest_FOUND TRUE)
else(${GTEST_FOUND} AND TARGET Threads::Threads)
  find_path(GTEST_INCLUDE_DIR gtest/gtest.h)
  find_path(GMOCK_INCLUDE_DIR gmock/gmock.h)
  find_library(GMOCK_LIBRARIES libgmock.so)
  find_library(GMOCK_MAIN_LIBRARIES libgmock_main.so)

  if(GTEST_INCLUDE_DIR AND GMOCK_INCLUDE_DIR AND GMOCK_LIBRARIES AND GMOCK_MAIN_LIBRARIES AND TARGET Threads::Threads)
    if(NOT TARGET gmock)
      add_library(gmock INTERFACE)
      target_include_directories(gmock INTERFACE ${GMOCK_INCLUDE_DIRS})
      target_link_libraries(gmock INTERFACE ${GMOCK_LIBRARIES} Threads::Threads)
    endif(NOT TARGET gmock)

    if(NOT TARGET gmock_main)
      add_library(gmock_main INTERFACE)
      target_include_directories(gmock_main INTERFACE ${GMOCK_INCLUDE_DIRS})
      target_link_libraries(gmock_main INTERFACE gmock)
      target_link_libraries(gmock_main INTERFACE ${GMOCK_MAIN_LIBRARIES})
    endif(NOT TARGET gmock_main)

    if(NOT TARGET gtest)
      add_library(gtest INTERFACE)
      target_include_directories(gtest INTERFACE ${GTEST_INCLUDE_DIRS})
      target_link_libraries(gtest INTERFACE ${GMOCK_LIBRARIES} Threads::Threads)
    endif(NOT TARGET gtest)

    if(NOT TARGET gtest_main)
      add_library(gtest_main INTERFACE)
      target_include_directories(gtest_main INTERFACE ${GTEST_INCLUDE_DIRS})
      target_link_libraries(gtest_main INTERFACE gtest)
      target_link_libraries(gtest_main INTERFACE ${GMOCK_MAIN_LIBRARIES})
    endif(NOT TARGET gtest_main)

    set(GTest_FOUND TRUE)
  endif(GTEST_INCLUDE_DIR AND GMOCK_INCLUDE_DIR AND GMOCK_LIBRARIES AND GMOCK_MAIN_LIBRARIES AND TARGET Threads::Threads)
endif(${GTEST_FOUND} AND TARGET Threads::Threads)
