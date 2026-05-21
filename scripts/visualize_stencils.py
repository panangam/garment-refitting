from __future__ import annotations

import numpy as np
import polyscope as ps
import torch

from example_data import load_default_test_pair
from refitting.binding import closest_points_on_mesh
from refitting.stencil import construct_garment_stencils


def main() -> None:
    (
        source_body_vertices,
        source_body_faces,
        source_garment_vertices,
        source_garment_faces,
        _target_body_vertices,
        _target_body_faces,
    ) = load_default_test_pair()

    binding = closest_points_on_mesh(
        source_garment_vertices,
        source_body_vertices,
        source_body_faces,
    )
    stencils = construct_garment_stencils(source_garment_faces, binding.face_ids)

    stencil_sizes = torch.tensor([stencil.numel() for stencil in stencils], dtype=torch.int32)
    sorted_by_size = torch.argsort(stencil_sizes)
    selected_vertex_ids = [
        int(sorted_by_size[0]),
        int(sorted_by_size[len(sorted_by_size) // 2]),
        int(sorted_by_size[-1]),
    ]

    source_body_vertices_np = source_body_vertices.numpy()
    source_body_faces_np = source_body_faces.numpy()
    source_garment_vertices_np = source_garment_vertices.numpy()
    source_garment_faces_np = source_garment_faces.numpy()

    scene_points = np.vstack([source_body_vertices_np, source_garment_vertices_np])
    bbox_low = scene_points.min(axis=0)
    bbox_high = scene_points.max(axis=0)
    length_scale = np.linalg.norm(bbox_high - bbox_low)

    # Use explicit centimeter units from the data, so point radii are not scene-relative.
    ps.set_autoscale_structures(False)
    ps.init()
    ps.set_automatically_compute_scene_extents(False)
    ps.set_bounding_box(bbox_low, bbox_high)
    ps.set_length_scale(length_scale)

    ps.register_surface_mesh("source body", source_body_vertices_np, source_body_faces_np)
    ps.register_surface_mesh(
        "source garment",
        source_garment_vertices_np,
        source_garment_faces_np,
        transparency=0.45,
    )

    for vertex_id in selected_vertex_ids:
        center = ps.register_point_cloud(
            f"center {vertex_id}",
            source_garment_vertices_np[vertex_id : vertex_id + 1],
        )
        center.set_radius(0.2, relative=False)

        stencil_vertices = source_garment_vertices_np[stencils[vertex_id].to(torch.long).numpy()]
        stencil_points = ps.register_point_cloud(
            f"stencil {vertex_id} ({stencils[vertex_id].numel()} vertices)",
            stencil_vertices,
        )
        stencil_points.set_radius(0.2, relative=False)

    ps.show()


if __name__ == "__main__":
    main()
