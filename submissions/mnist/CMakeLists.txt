cmake_minimum_required(VERSION 3.14)
project(add_lenet5 LANGUAGES CXX)

# ---------------------------------------------------------------
# 1.  Compiler / flags
# --------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
option( BUILD_STATIC "Set to ON to include static versions of the library" OFF)

# --------------------------------------------------------------------
# 2.  Find the OpenFHE *package* that the install script put into
#     third_party/openfhe
# --------------------------------------------------------------------
#   scripts/get_openfhe.sh installs:
#     third_party/openfhe/lib/cmake/OpenFHE/OpenFHEConfig.cmake

find_package(HEXL CONFIG QUIET)

find_package(OpenFHE CONFIG REQUIRED)


if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")

    if(TARGET HEXL::hexl)
        message(STATUS "HEXL target found.")
    else()
        message(STATUS "HEXL target NOT found.")
    endif()

else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

set( CMAKE_CXX_FLAGS "${OpenFHE_CXX_FLAGS} -Werror")

# --------------------------------------------------------------------
# 3.  Link libraries
# --------------------------------------------------------------------

include_directories( ${OPENMP_INCLUDES} )
include_directories( ${OpenFHE_INCLUDE} )
include_directories( ${OpenFHE_INCLUDE}/third-party/include )
include_directories( ${OpenFHE_INCLUDE}/core )
include_directories( ${OpenFHE_INCLUDE}/pke )
include_directories( ${OpenFHE_INCLUDE}/binfhe )
### add directories for other OpenFHE modules as needed for your project

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/fheon)

link_directories( ${OpenFHE_LIBDIR} )
link_directories( ${OPENMP_LIBRARIES} )

add_link_options(-Wl,--no-as-needed)

# Use classic RPATH (DT_RPATH) instead of the linker's default RUNPATH (DT_RUNPATH).
# RPATH is searched before LD_LIBRARY_PATH, so these executables always find this
# project's own OpenFHE build even if a different OpenFHE version's lib directory
# (e.g. /usr/local/lib) is present in LD_LIBRARY_PATH.
add_link_options(-Wl,--disable-new-dtags)

if(BUILD_STATIC)
    set( CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries( ${OpenFHE_STATIC_LIBRARIES} )
    message(STATUS "Build with static libs")
else()
    set( CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS} )
    link_libraries( ${OpenFHE_SHARED_LIBRARIES} )
endif()

if(TARGET HEXL::hexl)
    link_libraries(HEXL::hexl)
endif()


# The security level of the CKKS context (128 bits or performance efficient) is set by performanceMode in
# include/lenet5_fheon.h

# --------------------------------------------------------------------
# 4.  Create FHEON Libraries (HE controller and diagonal ANN controller)
# --------------------------------------------------------------------
add_library( fheonhecontroller fheon/FHEONHEController.cpp )
add_library( fheonannstandardcontroller fheon/FHEONANNStandardController.cpp )
target_link_libraries( fheonannstandardcontroller fheonhecontroller )
add_library( fheonanndiagonalcontroller fheon/FHEONANNDiagonalController.cpp )
target_link_libraries( fheonanndiagonalcontroller fheonannstandardcontroller )

add_library( encryption_utils src/encryption_utils.cpp )

# The LeNet-5 model (architecture, layouts, levels and layers) shared by the client and the server
add_library( lenet5_fheon src/lenet5_fheon.cpp )
target_link_libraries( lenet5_fheon fheonanndiagonalcontroller )

# --------------------------------------------------------------------
# 5.  Each *.cpp file becomes its own executable.
#     The seven stage names are hard-wired by the benchmark contract.
# --------------------------------------------------------------------

add_executable( client_key_generation src/client_key_generation.cpp )
target_link_libraries( client_key_generation lenet5_fheon )

add_executable( client_preprocess_input src/client_preprocess_input.cpp )

add_executable( client_encode_encrypt_input src/client_encode_encrypt_input.cpp )
target_link_libraries( client_encode_encrypt_input encryption_utils lenet5_fheon )

add_executable( client_decrypt_decode src/client_decrypt_decode.cpp )
target_link_libraries( client_decrypt_decode encryption_utils )

add_executable( client_postprocess src/client_postprocess.cpp )

add_executable( server_preprocess_model src/server_preprocess_model.cpp )

add_executable( server_encrypted_compute src/server_encrypted_compute.cpp )
target_link_libraries( server_encrypted_compute encryption_utils lenet5_fheon )
target_compile_definitions( server_encrypted_compute PRIVATE WEIGHTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/weights/lenet5/" )
