from __future__ import annotations

import torch


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
