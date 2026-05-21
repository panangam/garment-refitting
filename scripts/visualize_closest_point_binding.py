from __future__ import annotations

import numpy as np
import polyscope as ps

from example_data import load_default_test_pair
from refitting.binding import closest_points_on_mesh


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

    source_body_vertices_np = source_body_vertices.numpy()
    source_body_faces_np = source_body_faces.numpy()
    source_garment_vertices_np = source_garment_vertices.numpy()
    source_garment_faces_np = source_garment_faces.numpy()
    closest_points_np = binding.closest_points.numpy()

    scene_points = np.vstack([source_body_vertices_np, source_garment_vertices_np])
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
    ps.register_surface_mesh(
        "source garment",
        source_garment_vertices_np,
        source_garment_faces_np,
        transparency=0.5,
    )
    closest_points = ps.register_point_cloud("closest body points", closest_points_np)
    closest_points.set_radius(0.1, relative=False)

    segments = source_garment_vertices_np.repeat(2, axis=0)
    segments[1::2] = closest_points_np
    edges = np.arange(source_garment_vertices_np.shape[0] * 2, dtype=np.int32).reshape(-1, 2)
    offsets = ps.register_curve_network("closest-point offsets", segments, edges)
    offsets.set_radius(0.03, relative=False)
    ps.show()


if __name__ == "__main__":
    main()
