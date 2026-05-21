import torch

from refitting.initial_warp import compute_initial_warp, rotate_vectors_between_normals


def test_compute_initial_warp_identical_source_target_returns_original_garment():
    """Checks the initial warp is an identity when source and target bodies are identical."""
    body_vertices, body_faces = _source_body_mesh()
    garment_vertices = torch.tensor(
        [
            [0.2, 0.2, 0.5],
            [0.7, 0.2, 0.25],
            [0.2, 0.7, -0.3],
        ],
        dtype=torch.float32,
    )

    initial_warp = compute_initial_warp(
        garment_vertices,
        body_vertices,
        body_faces,
        body_vertices,
        body_faces,
    )

    torch.testing.assert_close(
        initial_warp.candidate_vertices,
        garment_vertices,
        atol=1e-6,
        rtol=0.0,
    )


def test_compute_initial_warp_preserves_displacement_lengths():
    """Verifies normal-aligned rotation preserves source displacement vector lengths."""
    source_body_vertices, source_body_faces = _source_body_mesh()
    target_body_vertices, target_body_faces = _target_body_mesh_rotated_about_x()
    garment_vertices = torch.tensor(
        [
            [0.2, 0.2, 0.5],
            [0.7, 0.2, 0.25],
            [0.2, 0.7, -0.3],
        ],
        dtype=torch.float32,
    )

    initial_warp = compute_initial_warp(
        garment_vertices,
        source_body_vertices,
        source_body_faces,
        target_body_vertices,
        target_body_faces,
    )
    source_displacements = garment_vertices - initial_warp.source_binding.closest_points

    torch.testing.assert_close(
        torch.linalg.norm(initial_warp.reoriented_displacements, dim=1),
        torch.linalg.norm(source_displacements, dim=1),
        atol=1e-6,
        rtol=0.0,
    )


def test_rotate_vectors_between_normals_handles_quarter_turn():
    """Checks the normal-aligned rotation maps vectors through a simple 90 degree turn."""
    vectors = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)
    source_normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    target_normals = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float32)

    rotated = rotate_vectors_between_normals(vectors, source_normals, target_normals)

    torch.testing.assert_close(
        rotated,
        torch.tensor([[0.0, -2.0, 0.0]], dtype=torch.float32),
        atol=1e-6,
        rtol=0.0,
    )


def _source_body_mesh():
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    return vertices, faces


def _target_body_mesh_rotated_about_x():
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    return vertices, faces
