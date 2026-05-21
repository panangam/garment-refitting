from __future__ import annotations

from dataclasses import dataclass

import torch

from refitting.binding import Binding, closest_points_on_mesh
from refitting.initial_warp import rotate_vectors_between_normals


@dataclass(frozen=True)
class Rebinding:
    target_binding: Binding
    reoriented_displacements: torch.Tensor
    candidate_vertices: torch.Tensor


def rebind_candidates_normal_aligned(
    relaxed_vertices: torch.Tensor,
    target_body_vertices: torch.Tensor,
    target_body_faces: torch.Tensor,
    source_closest_points: torch.Tensor,
    source_normals: torch.Tensor,
    original_garment_vertices: torch.Tensor,
) -> Rebinding:
    target_binding = closest_points_on_mesh(
        relaxed_vertices,
        target_body_vertices,
        target_body_faces,
    )
    source_displacements = original_garment_vertices - source_closest_points
    reoriented_displacements = rotate_vectors_between_normals(
        source_displacements,
        source_normals,
        target_binding.normals,
    )
    return Rebinding(
        target_binding=target_binding,
        reoriented_displacements=reoriented_displacements,
        candidate_vertices=target_binding.closest_points + reoriented_displacements,
    )
