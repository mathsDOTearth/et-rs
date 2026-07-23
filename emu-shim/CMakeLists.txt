# Builds the C ABI shim over the device-layer software-emulator backend as a
# shared library. Using the SDK's own find_package configuration resolves the
# full transitive link chain (sw-sysemu, hostUtils, linuxDriver, Boost, glog,
# lz4) that the Rust build then need not reconstruct by hand.
#
#   cmake -B build -DCMAKE_PREFIX_PATH=/opt/et
#   cmake --build build

cmake_minimum_required(VERSION 3.17)
project(et_emu_shim LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_PREFIX_PATH}/lib/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_PREFIX_PATH}/lib/cmake/cmake-modules")

find_package(deviceLayer REQUIRED)
find_package(sw-sysemu REQUIRED)

add_library(et_emu_shim SHARED et_emu_shim.cpp)
target_compile_features(et_emu_shim PRIVATE cxx_std_17)
target_include_directories(et_emu_shim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(et_emu_shim
    PRIVATE
        deviceLayer::deviceLayer
        sw-sysemu::sw-sysemu
)

install(TARGETS et_emu_shim LIBRARY DESTINATION lib)
