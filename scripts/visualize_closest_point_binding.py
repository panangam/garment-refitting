from __future__ import annotations

import numpy as np
import polyscope as ps

from refitting.binding import closest_points_on_mesh


def main() -> None:
    body_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    body_faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    query_points = np.array(
        [
            [0.20, 0.20, 0.35],
            [0.75, 0.25, -0.25],
            [1.25, 0.50, 0.15],
            [-0.15, 0.40, 0.20],
        ],
        dtype=np.float32,
    )

    binding = closest_points_on_mesh(query_points, body_vertices, body_faces)

    ps.init()
    ps.register_surface_mesh("body", body_vertices, body_faces)
    ps.register_point_cloud("query points", query_points, radius=0.02)
    ps.register_point_cloud("closest points", binding.closest_points, radius=0.02)

    segments = np.empty((query_points.shape[0] * 2, 3), dtype=np.float32)
    segments[0::2] = query_points
    segments[1::2] = binding.closest_points
    edges = np.arange(query_points.shape[0] * 2, dtype=np.int32).reshape(-1, 2)
    ps.register_curve_network("closest-point offsets", segments, edges, radius=0.005)
    ps.show()


if __name__ == "__main__":
    main()
