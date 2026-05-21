import torch

from refitting.initial_warp import compute_initial_warp
from refitting.rebinding import rebind_candidates_normal_aligned


def test_rebind_candidates_normal_aligned_outputs_finite_candidates():
    """Checks rebinding returns finite next candidates with the garment vertex shape."""
    body_vertices, body_faces = _body_mesh()
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

    rebinding = rebind_candidates_normal_aligned(
        initial_warp.candidate_vertices,
        body_vertices,
        body_faces,
        initial_warp.source_binding.closest_points,
        initial_warp.source_binding.normals,
        garment_vertices,
    )

    assert rebinding.candidate_vertices.shape == garment_vertices.shape
    assert bool(torch.isfinite(rebinding.candidate_vertices).all())
    assert rebinding.target_binding.closest_points.shape == garment_vertices.shape


def test_rebind_candidates_normal_aligned_is_stable_when_binding_does_not_change():
    """Verifies candidates remain stable if closest points and normals are unchanged."""
    body_vertices, body_faces = _body_mesh()
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

    rebinding = rebind_candidates_normal_aligned(
        initial_warp.candidate_vertices,
        body_vertices,
        body_faces,
        initial_warp.source_binding.closest_points,
        initial_warp.source_binding.normals,
        garment_vertices,
    )

    torch.testing.assert_close(
        rebinding.candidate_vertices,
        initial_warp.candidate_vertices,
        atol=1e-6,
        rtol=0.0,
    )


def _body_mesh():
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
