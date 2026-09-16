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

        # Allow the user to point to a ROSE install via environment or cache.
        if(NOT DEFINED ROSE_INSTALL_PREFIX)
            set(ROSE_INSTALL_PREFIX "$ENV{ROSE_INSTALL_PREFIX}")
        endif()
        if(NOT ROSE_INSTALL_PREFIX)
            # Try a few common defaults.
            if(EXISTS "/home/superboy/Downloads/MadhurKumar004/loomX/install/lib/librose.so")
                set(ROSE_INSTALL_PREFIX "/home/superboy/Downloads/MadhurKumar004/loomX/install")
            elseif(EXISTS "/usr/local/lib/librose.so")
                set(ROSE_INSTALL_PREFIX "/usr/local")
            elseif(EXISTS "/usr/lib/librose.so")
                set(ROSE_INSTALL_PREFIX "/usr")
            endif()
        endif()
        if(NOT ROSE_INSTALL_PREFIX)
            message(FATAL_ERROR "Please set ROSE_INSTALL_PREFIX to your ROSE installation directory")
        endif()

        list(APPEND CMAKE_PREFIX_PATH "${ROSE_INSTALL_PREFIX}")
        find_package(Rose QUIET)

        if(NOT TARGET Rose::rose)
            # No RoseConfig.cmake: fall back to manual include/library discovery.
            message(STATUS "loomX: RoseConfig.cmake not found; using manual ROSE paths")
            add_library(Rose::rose INTERFACE IMPORTED)
            set(ROSE_INCLUDE_DIRS "${ROSE_INSTALL_PREFIX}/include/rose")
            find_library(ROSE_LIBRARY rose PATHS "${ROSE_INSTALL_PREFIX}/lib" REQUIRED)
            target_include_directories(Rose::rose INTERFACE ${ROSE_INCLUDE_DIRS})
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

        set(CMAKE_INSTALL_RPATH "${ROSE_INSTALL_PREFIX}/lib")
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
