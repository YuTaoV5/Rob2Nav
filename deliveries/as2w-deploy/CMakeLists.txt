cmake_minimum_required(VERSION 3.16)
project(as2w_deploy LANGUAGES CXX)

add_library(as2w_policy policy.cpp)
target_compile_features(as2w_policy PUBLIC cxx_std_17)
target_include_directories(as2w_policy PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
if(MSVC)
    target_compile_options(as2w_policy PRIVATE /W4)
else()
    target_compile_options(as2w_policy PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(as2w_deploy main.cpp)
target_link_libraries(as2w_deploy PRIVATE as2w_policy)
include(CTest)
if(BUILD_TESTING)
    foreach(name recovery-r09 recovery-r12 recovery-r13 stairs-r14 stairs-r17-2000 stairs-r17-3000)
        add_test(NAME "${name}" COMMAND as2w_deploy verify
                 "${CMAKE_CURRENT_SOURCE_DIR}/models/${name}.bin"
                 "${CMAKE_CURRENT_SOURCE_DIR}/models/${name}.fixture")
    endforeach()
    add_executable(as2w_controller_test controller_test.cpp)
    target_link_libraries(as2w_controller_test PRIVATE as2w_policy)
    add_test(NAME controller COMMAND as2w_controller_test
             "${CMAKE_CURRENT_SOURCE_DIR}/models/recovery-r12.bin"
             "${CMAKE_CURRENT_SOURCE_DIR}/models/stairs-r17-3000.bin")
    set(students "${CMAKE_CURRENT_SOURCE_DIR}/recovery-students/selected")
    foreach(name recovery-r09-no-velocity recovery-r12-no-velocity recovery-r13-no-velocity
                 r12-alternate-delivery/recovery-r12-no-velocity-alt100)
        get_filename_component(label "${name}" NAME)
        add_test(NAME "${label}" COMMAND as2w_deploy verify
                 "${students}/${name}.bin" "${students}/${name}.fixture")
        add_test(NAME "controller-${label}" COMMAND as2w_controller_test
                 "${CMAKE_CURRENT_SOURCE_DIR}/models/recovery-r12.bin"
                 "${CMAKE_CURRENT_SOURCE_DIR}/models/stairs-r17-3000.bin"
                 "${students}/${name}.bin")
    endforeach()
endif()

option(AS2W_SDK2_SHADOW "Build the read-only SDK2 DDS probe" OFF)
if(AS2W_SDK2_SHADOW)
    find_package(unitree_sdk2 CONFIG REQUIRED)
    find_package(Threads REQUIRED)
    add_executable(as2w_sdk2_shadow sdk2_shadow.cpp)
    target_link_libraries(as2w_sdk2_shadow PRIVATE as2w_policy unitree_sdk2 Threads::Threads)
endif()
