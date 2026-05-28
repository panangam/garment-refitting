#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace nb = nanobind;

using VerticesTensor = nb::ndarray<nb::pytorch, const float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>;
using FacesTensor = nb::ndarray<nb::pytorch, const int32_t, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>;
using FieldTensor = nb::ndarray<nb::pytorch, float, nb::shape<-1, 2>, nb::device::cpu, nb::c_contig>;
using FaceTangentBasisTensor = nb::ndarray<nb::pytorch, float, nb::shape<-1, 2, 3>, nb::device::cpu, nb::c_contig>;

std::tuple<
    std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh>,
    std::unique_ptr<geometrycentral::surface::VertexPositionGeometry>>
make_geometry(VerticesTensor vertices, FacesTensor faces) {
  const size_t n_vertices = vertices.shape(0);
  const size_t n_faces = faces.shape(0);

  std::vector<geometrycentral::Vector3> vertex_positions(n_vertices);
  for (size_t i = 0; i < n_vertices; ++i) {
    vertex_positions[i] = geometrycentral::Vector3{
        static_cast<double>(vertices(i, 0)),
        static_cast<double>(vertices(i, 1)),
        static_cast<double>(vertices(i, 2)),
    };
  }

  std::vector<std::vector<size_t>> polygons(n_faces, std::vector<size_t>(3));
  for (size_t i = 0; i < n_faces; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      const int32_t index = faces(i, j);
      if (index < 0 || static_cast<size_t>(index) >= n_vertices) {
        throw std::out_of_range("face index is outside the vertex array");
      }
      polygons[i][j] = static_cast<size_t>(index);
    }
  }

  return geometrycentral::surface::makeManifoldSurfaceMeshAndGeometry(polygons, vertex_positions);
}

FieldTensor compute_curvature_aligned_face_direction_field(
    VerticesTensor vertices,
    FacesTensor faces,
    int n_sym = 2
) {
  if (n_sym != 2 && n_sym != 4) {
    throw std::logic_error("curvature-aligned direction fields only support n_sym = 2 or 4");
  }

  const size_t n_faces = faces.shape(0);
  std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh> mesh;
  std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = make_geometry(vertices, faces);

  geometrycentral::surface::FaceData<geometrycentral::Vector2> field =
      geometrycentral::surface::computeCurvatureAlignedFaceDirectionField(*geometry, n_sym);

  float* output = new float[n_faces * 2];
  for (size_t i = 0; i < n_faces; ++i) {
    const geometrycentral::Vector2 direction = field[mesh->face(i)];
    output[2 * i] = static_cast<float>(direction.x);
    output[2 * i + 1] = static_cast<float>(direction.y);
  }

  nb::capsule owner(output, [](void* data) noexcept {
    delete[] static_cast<float*>(data);
  });
  return FieldTensor(output, {n_faces, size_t(2)}, owner);
}

FaceTangentBasisTensor compute_face_tangent_basis(VerticesTensor vertices, FacesTensor faces) {
  const size_t n_faces = faces.shape(0);
  std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh> mesh;
  std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = make_geometry(vertices, faces);

  geometry->requireFaceTangentBasis();

  float* output = new float[n_faces * 2 * 3];
  for (size_t i = 0; i < n_faces; ++i) {
    const std::array<geometrycentral::Vector3, 2> basis = geometry->faceTangentBasis[mesh->face(i)];
    output[6 * i] = static_cast<float>(basis[0].x);
    output[6 * i + 1] = static_cast<float>(basis[0].y);
    output[6 * i + 2] = static_cast<float>(basis[0].z);
    output[6 * i + 3] = static_cast<float>(basis[1].x);
    output[6 * i + 4] = static_cast<float>(basis[1].y);
    output[6 * i + 5] = static_cast<float>(basis[1].z);
  }

  nb::capsule owner(output, [](void* data) noexcept {
    delete[] static_cast<float*>(data);
  });
  return FaceTangentBasisTensor(output, {n_faces, size_t(2), size_t(3)}, owner);
}

NB_MODULE(_geometry_central, module) {
  module.def(
      "compute_curvature_aligned_face_direction_field",
      &compute_curvature_aligned_face_direction_field,
      nb::arg("vertices"),
      nb::arg("faces"),
      nb::arg("n_sym") = 2,
      "Compute geometry-central's curvature-aligned face direction field."
  );
  module.def(
      "compute_face_tangent_basis",
      &compute_face_tangent_basis,
      nb::arg("vertices"),
      nb::arg("faces"),
      "Compute geometry-central's 3D face tangent basis."
  );
}
