from __future__ import annotations

import torch


def reconstruct_from_barycentric(
    body_vertices: torch.Tensor,
    body_faces: torch.Tensor,
    face_ids: torch.Tensor,
    barycentric_coords: torch.Tensor,
) -> torch.Tensor:
    triangles = body_vertices[body_faces.to(torch.long)[face_ids.to(torch.long)]]
    return torch.sum(barycentric_coords[:, :, None] * triangles, dim=1)
