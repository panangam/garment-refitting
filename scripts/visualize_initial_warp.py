from __future__ import annotations

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from example_data import load_default_test_pair
from refitting.initial_warp import compute_initial_warp


def main() -> None:
    (
        source_body_vertices,
        source_body_faces,
        source_garment_vertices,
        source_garment_faces,
        target_body_vertices,
        target_body_faces,
    ) = load_default_test_pair()

    initial_warp = compute_initial_warp(
        source_garment_vertices,
        source_body_vertices,
        source_body_faces,
        target_body_vertices,
        target_body_faces,
    )

    source_body_vertices_np = source_body_vertices.numpy()
    source_body_faces_np = source_body_faces.numpy()
    source_garment_vertices_np = source_garment_vertices.numpy()
    source_garment_faces_np = source_garment_faces.numpy()
    target_body_vertices_np = target_body_vertices.numpy()
    target_body_faces_np = target_body_faces.numpy()
    source_closest_points_np = initial_warp.source_binding.closest_points.numpy()
    target_anchor_points_np = initial_warp.target_anchor_points.numpy()
    candidate_vertices_np = initial_warp.candidate_vertices.numpy()

    source_points = np.vstack([source_body_vertices_np, source_garment_vertices_np])
    target_points = np.vstack([target_body_vertices_np, candidate_vertices_np])
    source_width = source_points[:, 0].max() - source_points[:, 0].min()
    target_width = target_points[:, 0].max() - target_points[:, 0].min()
    x_offset = source_width + 10.0
    target_shift = np.array([x_offset, 0.0, 0.0], dtype=np.float32)

    shifted_target_body_vertices_np = target_body_vertices_np + target_shift
    shifted_target_anchor_points_np = target_anchor_points_np + target_shift
    shifted_candidate_vertices_np = candidate_vertices_np + target_shift

    scene_points = np.vstack(
        [
            source_body_vertices_np,
            source_garment_vertices_np,
            shifted_target_body_vertices_np,
            shifted_candidate_vertices_np,
        ]
    )
    bbox_low = scene_points.min(axis=0)
    bbox_high = scene_points.max(axis=0)
    length_scale = np.linalg.norm(bbox_high - bbox_low)

    # Use explicit centimeter units from the data, so point/curve radii are not scene-relative.
    ps.set_autoscale_structures(False)
    ps.init()
    ps.set_automatically_compute_scene_extents(False)
    ps.set_bounding_box(bbox_low, bbox_high)
    ps.set_length_scale(length_scale)

    ps.register_surface_mesh("source body", source_body_vertices_np, source_body_faces_np)
    source_garment = ps.register_surface_mesh(
        "source garment",
        source_garment_vertices_np,
        source_garment_faces_np,
        transparency=0.45,
    )
    ps.register_surface_mesh(
        "target body",
        shifted_target_body_vertices_np,
        target_body_faces_np,
    )
    warped_garment = ps.register_surface_mesh(
        "initial warped garment",
        shifted_candidate_vertices_np,
        source_garment_faces_np,
        transparency=0.45,
    )

    source_closest_points = ps.register_point_cloud("source closest body points", source_closest_points_np)
    source_closest_points.set_radius(0.1, relative=False)

    source_segments = source_closest_points_np.repeat(2, axis=0)
    source_segments[1::2] = source_garment_vertices_np
    source_edges = np.arange(source_garment_vertices_np.shape[0] * 2, dtype=np.int32).reshape(-1, 2)
    source_displacements = ps.register_curve_network("source binding displacements", source_segments, source_edges)
    source_displacements.set_radius(0.03, relative=False)

    anchors = ps.register_point_cloud("target anchor points", shifted_target_anchor_points_np)
    anchors.set_radius(0.1, relative=False)

    segments = shifted_target_anchor_points_np.repeat(2, axis=0)
    segments[1::2] = shifted_candidate_vertices_np
    edges = np.arange(candidate_vertices_np.shape[0] * 2, dtype=np.int32).reshape(-1, 2)
    displacements = ps.register_curve_network("reoriented displacements", segments, edges)
    displacements.set_radius(0.03, relative=False)

    garment_translucent = True

    def callback() -> None:
        nonlocal garment_translucent
        changed, garment_translucent = psim.Checkbox("translucent garments", garment_translucent)
        if changed:
            transparency = 0.45 if garment_translucent else 1.0
            source_garment.set_transparency(transparency)
            warped_garment.set_transparency(transparency)

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
