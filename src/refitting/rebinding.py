from __future__ import annotations

from dataclasses import dataclass

import torch

from refitting.binding import Binding, closest_points_on_mesh
from refitting.directional_field import FaceFrameField
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


def rebind_candidates_directional_field(
    relaxed_vertices: torch.Tensor,
    target_body_vertices: torch.Tensor,
    target_body_faces: torch.Tensor,
    source_face_ids: torch.Tensor,
    target_anchor_displacements: torch.Tensor,
    target_face_frame_field: FaceFrameField,
) -> Rebinding:
    """Rebind using target-body face frames.

    ``source_face_ids`` are the initial source binding faces interpreted on the
    target mesh, which assumes source and target bodies share connectivity.
    """
    target_binding = closest_points_on_mesh(
        relaxed_vertices,
        target_body_vertices,
        target_body_faces,
    )
    reoriented_displacements = transport_displacements_with_directional_field(
        target_anchor_displacements,
        source_face_ids,
        target_binding.face_ids,
        target_face_frame_field,
    )
    return Rebinding(
        target_binding=target_binding,
        reoriented_displacements=reoriented_displacements,
        candidate_vertices=target_binding.closest_points + reoriented_displacements,
    )


def transport_displacements_with_directional_field(
    displacements: torch.Tensor,
    source_face_ids: torch.Tensor,
    target_face_ids: torch.Tensor,
    target_face_frame_field: FaceFrameField,
) -> torch.Tensor:
    source_frame = target_face_frame_field.frames[source_face_ids.to(torch.long)]
    target_frame = target_face_frame_field.frames[target_face_ids.to(torch.long)]

    components = torch.sum(displacements[:, None, :] * source_frame, dim=2)
    return torch.sum(components[:, :, None] * target_frame, dim=1)
