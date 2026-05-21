from __future__ import annotations

from dataclasses import dataclass

import torch

from refitting.binding import closest_points_on_mesh


@dataclass(frozen=True)
class AffineStencilWeights:
    stencils: list[torch.Tensor]
    weights: list[torch.Tensor]


def construct_affine_stencil_weights(
    garment_vertices: torch.Tensor,
    garment_faces: torch.Tensor,
    source_body_vertices: torch.Tensor,
    source_body_faces: torch.Tensor,
) -> AffineStencilWeights:
    binding = closest_points_on_mesh(
        garment_vertices,
        source_body_vertices,
        source_body_faces,
    )
    stencils = construct_garment_stencils(garment_faces, binding.face_ids)

    weights: list[torch.Tensor] = []
    for center, stencil in enumerate(stencils):
        stencil_vertices = garment_vertices[stencil.to(torch.long)]
        edges = (stencil_vertices - garment_vertices[center]).T
        covariance_inverse = torch.linalg.pinv(edges @ edges.T)
        weights.append(
            torch.eye(stencil.shape[0], dtype=torch.float32)
            - edges.T @ covariance_inverse @ edges
        )

    return AffineStencilWeights(stencils=stencils, weights=weights)


def construct_garment_stencils(
    garment_faces: torch.Tensor,
    closest_face_ids: torch.Tensor,
) -> list[torch.Tensor]:
    num_vertices = closest_face_ids.shape[0]
    stencils: list[set[int]] = [set() for _ in range(num_vertices)]

    for face in garment_faces.tolist():
        a, b, c = face
        stencils[a].update((b, c))
        stencils[b].update((a, c))
        stencils[c].update((a, b))

    face_clusters: dict[int, list[int]] = {}
    for vertex_id, face_id in enumerate(closest_face_ids.tolist()):
        face_clusters.setdefault(face_id, []).append(vertex_id)

    for cluster in face_clusters.values():
        for vertex_id in cluster:
            stencils[vertex_id].update(cluster)

    return [
        torch.tensor(sorted(vertex_ids - {center}), dtype=torch.int32)
        for center, vertex_ids in enumerate(stencils)
    ]
