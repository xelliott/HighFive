# Link against target system libs
# -------------------------------

# Independent target to make it possible to have new dependencies each build
add_library(libdeps INTERFACE)

if(NOT DEFINED HDF5_C_LIBRARIES)
  set(HDF5_NO_FIND_PACKAGE_CONFIG_FILE TRUE)  # Consistency
  set(HDF5_PREFER_PARALLEL ${HIGHFIVE_PARALLEL_HDF5})
  find_package(HDF5 REQUIRED)
endif()

if(HIGHFIVE_PARALLEL_HDF5 AND NOT HDF5_IS_PARALLEL)
  message(WARNING "Parallel HDF5 requested but libhdf5 doesnt support it")
endif()

target_link_libraries(libdeps INTERFACE HDF5::HDF5)
get_target_property(dep_def libdeps INTERFACE_COMPILE_DEFINITIONS)
if(NOT dep_def)
  set_target_properties(libdeps PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "")
endif()

# Boost
if(HIGHFIVE_USE_BOOST)
  set(Boost_NO_BOOST_CMAKE TRUE)  # Consistency
  find_package(Boost REQUIRED COMPONENTS system serialization)
  # Dont use imported targets yet, not avail before cmake 3.5
  target_include_directories(libdeps SYSTEM INTERFACE ${Boost_INCLUDE_DIR})
  target_compile_definitions(libdeps INTERFACE BOOST_ALL_NO_LIB H5_USE_BOOST)
endif()

# Eigen
if(HIGHFIVE_USE_EIGEN)
  if (NOT EIGEN3_INCLUDE_DIRS)
    find_package(Eigen3 NO_MODULE)
    if(Eigen3_FOUND)
      message(STATUS "Found Eigen ${Eigen3_VERSION}: ${EIGEN3_INCLUDE_DIRS}")
    else()
      find_package(PkgConfig)
      pkg_check_modules(EIGEN3 REQUIRED eigen3)
    endif()
  endif()
  if (NOT EIGEN3_INCLUDE_DIRS)
    message(FATAL_ERROR "Eigen was requested but could not be found")
  endif()
  target_include_directories(libdeps SYSTEM INTERFACE ${EIGEN3_INCLUDE_DIRS})
  target_compile_definitions(libdeps INTERFACE H5_USE_EIGEN)
endif()

# xtensor
if(HIGHFIVE_USE_XTENSOR)
  if (NOT xtensor_INCLUDE_DIRS)
    find_package(xtensor REQUIRED)
  endif()
  target_include_directories(libdeps SYSTEM INTERFACE ${xtensor_INCLUDE_DIRS})
  target_compile_definitions(libdeps INTERFACE H5_USE_XTENSOR)
endif()

# OpenCV
if(HIGHFIVE_USE_OPENCV)
  if (NOT OpenCV_INCLUDE_DIRS)
    find_package(OpenCV REQUIRED)
  endif()
  target_include_directories(libdeps SYSTEM INTERFACE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(libdeps INTERFACE ${OpenCV_LIBS})
  target_compile_definitions(libdeps INTERFACE H5_USE_OPENCV)
endif()

# MPI
if(HIGHFIVE_PARALLEL_HDF5 OR HDF5_IS_PARALLEL)
  find_package(MPI REQUIRED)
  target_link_libraries(libdeps INTERFACE MPI::MPI_CXX)
endif()

