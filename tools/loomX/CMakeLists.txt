cmake_minimum_required(VERSION 3.15)
project(loomX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect whether we are building inside the ROSE source tree or standalone.
if(TARGET Rose::rose)
    # In-tree build: Rose::rose is an alias for ROSE_DLL.
    message(STATUS "loomX: building inside ROSE source tree")
else()
    # Standalone build: require an installed ROSE.
    message(STATUS "loomX: building standalone against installed ROSE")
    set(CMAKE_PREFIX_PATH /home/kamini/projects/hackathon-rose/rose-install)
    set(BOOST_ROOT /home/kamini/projects/hackathon-rose/boost-install)
    find_package(Rose REQUIRED)

    # ROSE's CMake config asks for individual clang libraries, but this system
    # only provides the monolithic libclang-cpp.so.17. Redirect the clang
    # frontend target to the shared library.
    set(CLANG_CPP_LIB "/usr/lib/llvm-17/lib/libclang-cpp.so.17")
    if(NOT EXISTS "${CLANG_CPP_LIB}")
        message(FATAL_ERROR "Could not find ${CLANG_CPP_LIB}")
    endif()
    set_property(TARGET Rose::roseClangFrontend PROPERTY
        INTERFACE_LINK_LIBRARIES
        "${CLANG_CPP_LIB};Rose::rose_object_library_dependencies")

    # Set rpath so the binary finds librose.so at runtime
    set(CMAKE_INSTALL_RPATH
        "/home/kamini/projects/hackathon-rose/rose-install/lib;/home/kamini/projects/hackathon-rose/boost-install/lib;/usr/lib/llvm-17/lib")
endif()

# Compiler flags matching ROSE build environment
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem /usr/include/c++/15 -isystem /usr/include/x86_64-linux-gnu/c++/15")
set(CMAKE_C_COMPILER clang-17)
set(CMAKE_CXX_COMPILER clang++-17)

add_executable(loomX
    src/main.cpp
    src/InterproceduralAnalysis.cpp
    src/GpuProfitability.cpp
    src/OpenMPCodeGen.cpp
)

target_link_libraries(loomX PRIVATE Rose::rose)

if(NOT TARGET Rose::rose)
    set_target_properties(loomX PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_RPATH}"
        BUILD_WITH_INSTALL_RPATH TRUE)
endif()
