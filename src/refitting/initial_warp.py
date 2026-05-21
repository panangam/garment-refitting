from __future__ import annotations

from dataclasses import dataclass

import torch

from refitting.barycentric import reconstruct_from_barycentric
from refitting.binding import Binding, closest_points_on_mesh


@dataclass(frozen=True)
class InitialWarp:
    source_binding: Binding
    target_anchor_points: torch.Tensor
    target_normals: torch.Tensor
    reoriented_displacements: torch.Tensor
    candidate_vertices: torch.Tensor


def compute_initial_warp(
    garment_vertices: torch.Tensor,
    source_body_vertices: torch.Tensor,
    source_body_faces: torch.Tensor,
    target_body_vertices: torch.Tensor,
    target_body_faces: torch.Tensor,
) -> InitialWarp:
    source_binding = closest_points_on_mesh(
        garment_vertices,
        source_body_vertices,
        source_body_faces,
    )
    target_anchor_points = reconstruct_from_barycentric(
        target_body_vertices,
        target_body_faces,
        source_binding.face_ids,
        source_binding.barycentric_coords,
    )
    target_normals = _normals_at_faces(
        target_body_vertices,
        target_body_faces,
        source_binding.face_ids,
    )
    displacements = garment_vertices - source_binding.closest_points
    reoriented_displacements = rotate_vectors_between_normals(
        displacements,
        source_binding.normals,
        target_normals,
    )
    return InitialWarp(
        source_binding=source_binding,
        target_anchor_points=target_anchor_points,
        target_normals=target_normals,
        reoriented_displacements=reoriented_displacements,
        candidate_vertices=target_anchor_points + reoriented_displacements,
    )


def rotate_vectors_between_normals(
    vectors: torch.Tensor,
    source_normals: torch.Tensor,
    target_normals: torch.Tensor,
) -> torch.Tensor:
    cross_axes = torch.cross(source_normals, target_normals, dim=1)
    cos_angles = torch.sum(source_normals * target_normals, dim=1).clamp(-1.0, 1.0)
    rotated = torch.empty_like(vectors)

    same_normal = cos_angles > 1.0 - 1e-7
    opposite_normal = cos_angles < -1.0 + 1e-7
    general = ~(same_normal | opposite_normal)

    rotated[same_normal] = vectors[same_normal]
    rotated[general] = (
        vectors[general]
        + torch.cross(cross_axes[general], vectors[general], dim=1)
        + torch.cross(
            cross_axes[general],
            torch.cross(cross_axes[general], vectors[general], dim=1),
            dim=1,
        )
        / (1.0 + cos_angles[general])[:, None]
    )
    if bool(torch.any(opposite_normal)):
        axes = _orthogonal_unit_vectors(source_normals[opposite_normal])
        rotated[opposite_normal] = (
            -vectors[opposite_normal]
            + 2.0 * torch.sum(vectors[opposite_normal] * axes, dim=1)[:, None] * axes
        )

    return rotated


def _normals_at_faces(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_ids: torch.Tensor,
) -> torch.Tensor:
    triangles = vertices[faces.to(torch.long)[face_ids.to(torch.long)]]
    normals = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1)
    return normals / torch.linalg.norm(normals, dim=1, keepdim=True)


def _orthogonal_unit_vectors(normals: torch.Tensor) -> torch.Tensor:
    axes = torch.zeros_like(normals)
    use_x = torch.abs(normals[:, 0]) < 0.9
    axes[use_x, 0] = 1.0
    axes[~use_x, 1] = 1.0
    axes = axes - torch.sum(axes * normals, dim=1)[:, None] * normals
    return axes / torch.linalg.norm(axes, dim=1, keepdim=True)
