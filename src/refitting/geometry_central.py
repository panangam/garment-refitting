from __future__ import annotations

import torch

from refitting import _geometry_central


def compute_curvature_aligned_face_direction_field(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    n_sym: int = 2,
) -> torch.Tensor:
    """Compute geometry-central's curvature-aligned face direction field.

    The returned tensor has shape ``(num_faces, 2)``. For ``n_sym > 1``,
    geometry-central returns its symmetric field in power representation.
    """
    return _geometry_central.compute_curvature_aligned_face_direction_field(
        vertices.contiguous(),
        faces.contiguous(),
        n_sym,
    )


def compute_smoothest_face_direction_field(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    n_sym: int = 1,
) -> torch.Tensor:
    """Compute geometry-central's smoothest face direction field.

    The returned tensor has shape ``(num_faces, 2)``. For ``n_sym == 1``, this
    is a signed face-local direction field.
    """
    return _geometry_central.compute_smoothest_face_direction_field(
        vertices.contiguous(),
        faces.contiguous(),
        n_sym,
    )


def compute_face_tangent_basis(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Compute geometry-central's 3D face tangent basis.

    The returned tensor has shape ``(num_faces, 2, 3)``. The second dimension
    stores the local x and y tangent axes used by geometry-central for each face.
    """
    return _geometry_central.compute_face_tangent_basis(
        vertices.contiguous(),
        faces.contiguous(),
    )
