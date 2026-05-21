import torch

from refitting.binding import closest_points_on_mesh


def test_closest_points_on_mesh_binding_shapes_and_dtypes():
    """Checks Step 1 binding returns all expected fields with CPU tensor shapes and dtypes."""
    body_vertices, body_faces = _square_mesh()
    query_points = torch.tensor(
        [
            [0.25, 0.25, 0.50],
            [0.75, 0.25, -0.25],
            [1.20, 0.50, 0.10],
        ],
        dtype=torch.float32,
    )

    binding = closest_points_on_mesh(query_points, body_vertices, body_faces)

    assert binding.closest_points.shape == (3, 3)
    assert binding.closest_points.dtype == torch.float32
    assert binding.closest_points.device.type == "cpu"
    assert binding.face_ids.shape == (3,)
    assert binding.face_ids.dtype == torch.int32
    assert binding.barycentric_coords.shape == (3, 3)
    assert binding.barycentric_coords.dtype == torch.float32
    assert binding.distances_squared.shape == (3,)
    assert binding.distances_squared.dtype == torch.float32
    assert binding.normals.shape == (3, 3)
    assert binding.normals.dtype == torch.float32


def test_closest_points_on_mesh_binding_reconstructs_closest_points():
    """Verifies barycentric coordinates and face IDs reconstruct the returned closest points."""
    body_vertices, body_faces = _square_mesh()
    query_points = torch.tensor(
        [
            [0.20, 0.20, 0.35],
            [0.75, 0.25, -0.25],
            [1.25, 0.50, 0.15],
            [-0.15, 0.40, 0.20],
        ],
        dtype=torch.float32,
    )

    binding = closest_points_on_mesh(query_points, body_vertices, body_faces)
    triangles = body_vertices[body_faces.to(torch.long)[binding.face_ids.to(torch.long)]]
    reconstructed = torch.einsum("ni,nij->nj", binding.barycentric_coords, triangles)

    torch.testing.assert_close(reconstructed, binding.closest_points, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        binding.barycentric_coords.sum(dim=1),
        torch.ones(binding.barycentric_coords.shape[0]),
        atol=1e-6,
        rtol=0.0,
    )
    assert bool(torch.all(binding.face_ids >= 0))
    assert bool(torch.all(binding.face_ids < body_faces.shape[0]))


def test_closest_points_on_mesh_binding_normals_are_finite_unit_vectors():
    """Checks closest-triangle normals are finite unit vectors with the expected orientation."""
    body_vertices, body_faces = _square_mesh()
    query_points = torch.tensor(
        [
            [0.15, 0.15, 0.10],
            [0.85, 0.85, -0.20],
        ],
        dtype=torch.float32,
    )

    binding = closest_points_on_mesh(query_points, body_vertices, body_faces)
    normal_lengths = torch.linalg.norm(binding.normals, dim=1)

    assert bool(torch.isfinite(binding.normals).all())
    torch.testing.assert_close(normal_lengths, torch.ones(2), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        binding.normals,
        torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.float32),
    )


def _square_mesh():
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32)
    return vertices, faces
