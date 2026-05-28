from __future__ import annotations

import torch

from refitting.geometry_central import compute_curvature_aligned_face_direction_field, compute_face_tangent_basis


def test_compute_curvature_aligned_face_direction_field_runs() -> None:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=torch.int32,
    )

    field = compute_curvature_aligned_face_direction_field(vertices, faces)
    face_tangent_basis = compute_face_tangent_basis(vertices, faces)

    representative_angles = torch.atan2(field[:, 1], field[:, 0]) / 2
    local_directions = torch.stack(
        [torch.cos(representative_angles), torch.sin(representative_angles)],
        dim=1,
    )
    directions_3d = (
        local_directions[:, 0, None] * face_tangent_basis[:, 0, :]
        + local_directions[:, 1, None] * face_tangent_basis[:, 1, :]
    )

    assert field.shape == (faces.shape[0], 2)
    assert field.dtype == torch.float32
    assert field.device.type == "cpu"
    assert face_tangent_basis.shape == (faces.shape[0], 2, 3)
    assert face_tangent_basis.dtype == torch.float32
    assert face_tangent_basis.device.type == "cpu"
    assert directions_3d.shape == (faces.shape[0], 3)
    assert torch.allclose(torch.linalg.norm(face_tangent_basis, dim=2), torch.ones((faces.shape[0], 2)))
    assert torch.allclose(torch.sum(face_tangent_basis[:, 0, :] * face_tangent_basis[:, 1, :], dim=1), torch.zeros(faces.shape[0]))
