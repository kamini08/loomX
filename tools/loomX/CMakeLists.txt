# loomX CMakeLists.txt
# Builds both inside the ROSE source tree and as a standalone project.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # Standalone build
    cmake_minimum_required(VERSION 3.15)
    project(loomX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    if(NOT TARGET Rose::rose)
        message(STATUS "loomX: building standalone against installed ROSE")

        # The user can point CMake at a ROSE installation explicitly.
        #   cmake .. -DROSE_INSTALL_PREFIX=/path/to/rose/install
        # or via the environment:
        #   ROSE_INSTALL_PREFIX=/path/to/rose/install cmake ..
        if(NOT DEFINED ROSE_INSTALL_PREFIX OR ROSE_INSTALL_PREFIX STREQUAL "")
            set(ROSE_INSTALL_PREFIX "$ENV{ROSE_INSTALL_PREFIX}")
        endif()

        # If a prefix was given, prefer it for searching.
        if(ROSE_INSTALL_PREFIX)
            list(APPEND CMAKE_PREFIX_PATH "${ROSE_INSTALL_PREFIX}")
        endif()

        # Try CMake-config mode first (RoseConfig.cmake).
        find_package(Rose QUIET)

        if(NOT TARGET Rose::rose)
            # No RoseConfig.cmake: fall back to manual discovery.
            message(STATUS "loomX: RoseConfig.cmake not found; using manual ROSE discovery")

            if(ROSE_INSTALL_PREFIX)
                # Search only in the user-supplied prefix.
                find_path(ROSE_INCLUDE_DIR rose.h
                    PATHS "${ROSE_INSTALL_PREFIX}/include/rose"
                    NO_DEFAULT_PATH
                    REQUIRED)
                find_library(ROSE_LIBRARY rose
                    PATHS "${ROSE_INSTALL_PREFIX}/lib"
                    NO_DEFAULT_PATH
                    REQUIRED)
            else()
                # Search the system for an installed ROSE.
                find_path(ROSE_INCLUDE_DIR rose.h
                    PATH_SUFFIXES rose
                    REQUIRED)
                find_library(ROSE_LIBRARY rose REQUIRED)
            endif()

            message(STATUS "loomX: ROSE include dir: ${ROSE_INCLUDE_DIR}")
            message(STATUS "loomX: ROSE library:     ${ROSE_LIBRARY}")

            add_library(Rose::rose INTERFACE IMPORTED)
            target_include_directories(Rose::rose INTERFACE ${ROSE_INCLUDE_DIR})

            find_package(Boost REQUIRED COMPONENTS
                date_time thread filesystem program_options regex system
                serialization wave iostreams chrono atomic)

            target_link_libraries(Rose::rose INTERFACE
                ${ROSE_LIBRARY}
                Boost::date_time Boost::thread Boost::filesystem Boost::program_options
                Boost::regex Boost::system Boost::serialization Boost::wave
                Boost::iostreams Boost::chrono Boost::atomic
                dl pthread m)
        endif()

        # Set rpath so the binary finds librose.so at runtime.
        get_filename_component(ROSE_LIB_DIR "${ROSE_LIBRARY}" DIRECTORY)
        set(CMAKE_INSTALL_RPATH "${ROSE_LIB_DIR}")
    endif()

    add_executable(loomX
        src/main.cpp
        src/InterproceduralAnalysis.cpp
        src/GpuProfitability.cpp
        src/OpenMPCodeGen.cpp
        src/LoopCanonicalChecker.cpp
        src/IterationCountEstimator.cpp
        src/DivergenceAnalyzer.cpp
        src/ComputeIntensityEstimator.cpp
        src/ReductionDetector.cpp
    )
    target_include_directories(loomX PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(loomX PRIVATE Rose::rose)
    set_target_properties(loomX PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_RPATH}"
        BUILD_WITH_INSTALL_RPATH TRUE)
else()
    # In-tree build inside ROSE
    message(STATUS "loomX: building inside ROSE source tree")

    set(tools_loomX_exe_SOURCES
        src/main.cpp
        src/InterproceduralAnalysis.cpp
        src/GpuProfitability.cpp
        src/OpenMPCodeGen.cpp
        src/LoopCanonicalChecker.cpp
        src/IterationCountEstimator.cpp
        src/DivergenceAnalyzer.cpp
        src/ComputeIntensityEstimator.cpp
        src/ReductionDetector.cpp
    )

    add_executable(tools_loomX_exe ${tools_loomX_exe_SOURCES})
    target_link_libraries(tools_loomX_exe PRIVATE ROSE_DLL)
    target_include_directories(tools_loomX_exe
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                ${CMAKE_BINARY_DIR} ${ROSE_INCLUDES})

    set_target_properties(tools_loomX_exe PROPERTIES OUTPUT_NAME "loomX")

    install(TARGETS tools_loomX_exe DESTINATION bin)

    add_dependencies(tools tools_loomX_exe)
endif()
