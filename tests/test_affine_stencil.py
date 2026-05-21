import torch

from refitting.affine_stencil import construct_affine_stencil_weights


def test_construct_affine_stencil_weights_shapes_symmetry_and_idempotence():
    """Checks each W_v has the expected square shape and behaves as a projection matrix."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()

    affine_weights = construct_affine_stencil_weights(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
    )

    for stencil, weight in zip(affine_weights.stencils, affine_weights.weights, strict=True):
        assert weight.shape == (stencil.numel(), stencil.numel())
        torch.testing.assert_close(weight, weight.T, atol=1e-5, rtol=0.0)
        torch.testing.assert_close(weight @ weight, weight, atol=1e-5, rtol=0.0)


def test_construct_affine_stencil_weights_original_residual_is_near_zero():
    """Verifies the original stencil edges lie in the affine subspace removed by W_v."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()

    affine_weights = construct_affine_stencil_weights(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
    )

    for center, stencil, weight in zip(
        range(len(affine_weights.stencils)),
        affine_weights.stencils,
        affine_weights.weights,
        strict=True,
    ):
        edges = garment_vertices[stencil.to(torch.long)] - garment_vertices[center]
        residual = edges.T @ weight
        torch.testing.assert_close(
            residual,
            torch.zeros_like(residual),
            atol=1e-5,
            rtol=0.0,
        )


def test_construct_affine_stencil_weights_affine_transformed_residual_is_near_zero():
    """Checks W_v also removes residuals after an arbitrary affine transform of the stencil."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()
    transform = torch.tensor(
        [
            [1.2, 0.1, -0.2],
            [0.3, 0.8, 0.4],
            [-0.1, 0.2, 1.5],
        ],
        dtype=torch.float32,
    )
    translation = torch.tensor([3.0, -2.0, 0.5], dtype=torch.float32)
    transformed_vertices = garment_vertices @ transform.T + translation

    affine_weights = construct_affine_stencil_weights(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
    )

    for center, stencil, weight in zip(
        range(len(affine_weights.stencils)),
        affine_weights.stencils,
        affine_weights.weights,
        strict=True,
    ):
        edges = transformed_vertices[stencil.to(torch.long)] - transformed_vertices[center]
        residual = edges.T @ weight
        torch.testing.assert_close(
            residual,
            torch.zeros_like(residual),
            atol=1e-5,
            rtol=0.0,
        )


def _garment_and_body_meshes():
    garment_vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.5],
        ],
        dtype=torch.float32,
    )
    garment_faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
            [1, 2, 3],
            [1, 2, 4],
        ],
        dtype=torch.int32,
    )
    body_vertices = garment_vertices.clone()
    body_faces = garment_faces.clone()
    return garment_vertices, garment_faces, body_vertices, body_faces
