cmake_minimum_required(VERSION 3.18)

project(refitting LANGUAGES CXX)

include(GNUInstallDirs)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/geometry-central/cmake")

execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "from pathlib import Path; import site, sysconfig; paths = [sysconfig.get_path('purelib'), sysconfig.get_path('platlib'), *site.getsitepackages()]; candidates = [Path(p) / 'include' for p in paths if p]; print(next((str(p) for p in candidates if (p / 'Eigen/Core').exists()), ''), end='')"
  OUTPUT_VARIABLE REFITTING_EIGEN_INCLUDE
)
if(REFITTING_EIGEN_INCLUDE)
  add_library(refitting_eigen INTERFACE)
  add_library(Eigen3::Eigen ALIAS refitting_eigen)
  target_include_directories(refitting_eigen SYSTEM INTERFACE "${REFITTING_EIGEN_INCLUDE}")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
set(SUITESPARSE OFF CACHE BOOL "Enable SuiteSparse for geometry-central" FORCE)

set(GEOMETRY_CENTRAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/geometry-central")
add_library(geometry-central STATIC
  "${GEOMETRY_CENTRAL_ROOT}/src/surface/surface_mesh.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/surface/manifold_surface_mesh.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/surface/halfedge_factories.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/surface/surface_mesh_factories.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/surface/base_geometry_interface.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/surface/intrinsic_geometry_interface.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/surface/extrinsic_geometry_interface.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/surface/embedded_geometry_interface.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/surface/edge_length_geometry.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/surface/vertex_position_geometry.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/surface/direction_fields.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/numerical/linear_algebra_utilities.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/numerical/linear_solvers.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/numerical/eigenproblem_solvers.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/numerical/qr_solvers.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/numerical/square_solvers.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/numerical/positive_definite_solvers.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/utilities/utilities.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/utilities/quaternion.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/utilities/disjoint_sets.cpp"
  "${GEOMETRY_CENTRAL_ROOT}/src/utilities/elementary_geometry.cpp"
)
target_include_directories(geometry-central PUBLIC "${GEOMETRY_CENTRAL_ROOT}/include")
target_link_libraries(geometry-central PUBLIC Eigen3::Eigen)
target_compile_features(geometry-central PUBLIC cxx_std_11)
target_compile_definitions(geometry-central PUBLIC NOMINMAX _USE_MATH_DEFINES)
set_target_properties(geometry-central PROPERTIES POSITION_INDEPENDENT_CODE ON)

nanobind_add_module(_geometry_central src/cpp/geometry_central.cpp)
target_compile_features(_geometry_central PRIVATE cxx_std_17)
target_link_libraries(_geometry_central PRIVATE geometry-central)

install(TARGETS _geometry_central LIBRARY DESTINATION refitting)
