import numpy as np

from refitting.binding import closest_points_on_mesh


def test_closest_points_on_mesh_binding_shapes_and_dtypes():
    body_vertices, body_faces = _square_mesh()
    query_points = np.array(
        [
            [0.25, 0.25, 0.50],
            [0.75, 0.25, -0.25],
            [1.20, 0.50, 0.10],
        ],
        dtype=np.float32,
    )

    binding = closest_points_on_mesh(query_points, body_vertices, body_faces)

    assert binding.closest_points.shape == (3, 3)
    assert binding.closest_points.dtype == np.float32
    assert binding.face_ids.shape == (3,)
    assert binding.face_ids.dtype == np.int32
    assert binding.barycentric_coords.shape == (3, 3)
    assert binding.barycentric_coords.dtype == np.float32
    assert binding.distances_squared.shape == (3,)
    assert binding.distances_squared.dtype == np.float32
    assert binding.normals.shape == (3, 3)
    assert binding.normals.dtype == np.float32


def test_closest_points_on_mesh_binding_reconstructs_closest_points():
    body_vertices, body_faces = _square_mesh()
    query_points = np.array(
        [
            [0.20, 0.20, 0.35],
            [0.75, 0.25, -0.25],
            [1.25, 0.50, 0.15],
            [-0.15, 0.40, 0.20],
        ],
        dtype=np.float32,
    )

    binding = closest_points_on_mesh(query_points, body_vertices, body_faces)
    triangles = body_vertices[body_faces[binding.face_ids]]
    reconstructed = np.einsum("ni,nij->nj", binding.barycentric_coords, triangles)

    np.testing.assert_allclose(reconstructed, binding.closest_points, atol=1e-6)
    np.testing.assert_allclose(binding.barycentric_coords.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(binding.face_ids >= 0)
    assert np.all(binding.face_ids < body_faces.shape[0])


def test_closest_points_on_mesh_binding_normals_are_finite_unit_vectors():
    body_vertices, body_faces = _square_mesh()
    query_points = np.array(
        [
            [0.15, 0.15, 0.10],
            [0.85, 0.85, -0.20],
        ],
        dtype=np.float32,
    )

    binding = closest_points_on_mesh(query_points, body_vertices, body_faces)
    normal_lengths = np.linalg.norm(binding.normals, axis=1)

    assert np.isfinite(binding.normals).all()
    np.testing.assert_allclose(normal_lengths, 1.0, atol=1e-6)
    np.testing.assert_allclose(binding.normals, np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32))


def _square_mesh():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return vertices, faces
