import torch

from refitting.barycentric import reconstruct_from_barycentric
from refitting.binding import closest_points_on_mesh


def test_reconstruct_from_barycentric_known_triangle_points():
    body_vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=torch.float32,
    )
    body_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    face_ids = torch.zeros(6, dtype=torch.int32)
    barycentric_coords = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ],
        dtype=torch.float32,
    )

    points = reconstruct_from_barycentric(body_vertices, body_faces, face_ids, barycentric_coords)

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0 / 3.0, 4.0 / 3.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(points, expected, atol=1e-6, rtol=0.0)


def test_reconstruct_from_barycentric_reproduces_closest_point_binding():
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
    reconstructed = reconstruct_from_barycentric(
        body_vertices,
        body_faces,
        binding.face_ids,
        binding.barycentric_coords,
    )

    assert reconstructed.dtype == torch.float32
    assert reconstructed.device.type == "cpu"
    torch.testing.assert_close(reconstructed, binding.closest_points, atol=1e-6, rtol=0.0)


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
