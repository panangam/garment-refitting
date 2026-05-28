import torch

from refitting.directional_field import compute_face_frame_field


def test_compute_face_frame_field_returns_face_frame_data():
    vertices, faces = _tetrahedron_mesh()

    face_frame_field = compute_face_frame_field(vertices, faces)

    assert face_frame_field.field_2d.shape == (faces.shape[0], 2)
    assert face_frame_field.tangent_basis.shape == (faces.shape[0], 2, 3)
    assert face_frame_field.directions.shape == (faces.shape[0], 3)
    assert face_frame_field.normals.shape == (faces.shape[0], 3)
    assert face_frame_field.frames.shape == (faces.shape[0], 3, 3)
    assert torch.isfinite(face_frame_field.frames).all()
    torch.testing.assert_close(
        torch.linalg.norm(face_frame_field.frames, dim=2),
        torch.ones((faces.shape[0], 3)),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.sum(face_frame_field.frames[:, 0, :] * face_frame_field.frames[:, 1, :], dim=1),
        torch.zeros(faces.shape[0]),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.sum(face_frame_field.frames[:, 0, :] * face_frame_field.frames[:, 2, :], dim=1),
        torch.zeros(faces.shape[0]),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.sum(face_frame_field.frames[:, 1, :] * face_frame_field.frames[:, 2, :], dim=1),
        torch.zeros(faces.shape[0]),
        atol=1e-5,
        rtol=0.0,
    )


def _tetrahedron_mesh() -> tuple[torch.Tensor, torch.Tensor]:
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
    return vertices, faces
