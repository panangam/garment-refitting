from __future__ import annotations

from dataclasses import dataclass

import torch

from refitting.geometry_central import compute_face_tangent_basis, compute_smoothest_face_direction_field


@dataclass(frozen=True)
class FaceFrameField:
    field_2d: torch.Tensor
    tangent_basis: torch.Tensor
    directions: torch.Tensor
    normals: torch.Tensor
    frames: torch.Tensor


def compute_face_frame_field(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    n_sym: int = 1,
) -> FaceFrameField:
    field_2d = compute_smoothest_face_direction_field(vertices, faces, n_sym=n_sym)
    tangent_basis = compute_face_tangent_basis(vertices, faces)
    directions = field_2d[:, 0, None] * tangent_basis[:, 0, :] + field_2d[:, 1, None] * tangent_basis[:, 1, :]
    directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True)

    triangles = vertices[faces.to(torch.long)]
    normals = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1)
    normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True)

    perpendiculars = torch.cross(normals, directions, dim=1)
    perpendiculars = perpendiculars / torch.linalg.norm(perpendiculars, dim=1, keepdim=True)
    frames = torch.stack([directions, perpendiculars, normals], dim=1)

    return FaceFrameField(
        field_2d=field_2d,
        tangent_basis=tangent_basis,
        directions=directions,
        normals=normals,
        frames=frames,
    )
