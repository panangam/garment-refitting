from __future__ import annotations

from dataclasses import dataclass

import igl
import numpy as np


@dataclass(frozen=True)
class Binding:
    closest_points: np.ndarray
    face_ids: np.ndarray
    barycentric_coords: np.ndarray
    distances_squared: np.ndarray
    normals: np.ndarray


def closest_points_on_mesh(
    query_points: np.ndarray,
    body_vertices: np.ndarray,
    body_faces: np.ndarray,
) -> Binding:
    """Bind query points to their closest points on a triangle mesh."""
    query_points = np.asarray(query_points, dtype=np.float32)
    body_vertices = np.asarray(body_vertices, dtype=np.float32)
    body_faces = np.asarray(body_faces, dtype=np.int32)

    distances_squared, face_ids, closest_points = igl.point_mesh_squared_distance(
        query_points.astype(np.float64, copy=False),
        body_vertices.astype(np.float64, copy=False),
        body_faces,
    )

    face_ids = np.asarray(face_ids, dtype=np.int32)
    closest_points = np.asarray(closest_points, dtype=np.float32)
    distances_squared = np.asarray(distances_squared, dtype=np.float32)

    triangles = body_vertices[body_faces[face_ids]]
    barycentric_coords = _barycentric_coordinates(closest_points, triangles)
    normals = _triangle_normals(triangles)

    return Binding(
        closest_points=closest_points,
        face_ids=face_ids,
        barycentric_coords=barycentric_coords.astype(np.float32, copy=False),
        distances_squared=distances_squared,
        normals=normals.astype(np.float32, copy=False),
    )


def _barycentric_coordinates(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]

    v0 = b - a
    v1 = c - a
    v2 = points - a

    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)

    denom = d00 * d11 - d01 * d01
    assert np.all(np.abs(denom) > 1e-12), "Body mesh contains degenerate closest triangles."

    beta = (d11 * d20 - d01 * d21) / denom
    gamma = (d00 * d21 - d01 * d20) / denom
    alpha = 1.0 - beta - gamma
    return np.stack([alpha, beta, gamma], axis=1)


def _triangle_normals(triangles: np.ndarray) -> np.ndarray:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    assert np.all(lengths[:, 0] > 1e-12), "Body mesh contains degenerate closest triangles."
    return normals / lengths
