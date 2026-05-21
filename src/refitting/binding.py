from __future__ import annotations

from dataclasses import dataclass

import igl
import numpy as np
import torch


@dataclass(frozen=True)
class Binding:
    closest_points: torch.Tensor
    face_ids: torch.Tensor
    barycentric_coords: torch.Tensor
    distances_squared: torch.Tensor
    normals: torch.Tensor


def closest_points_on_mesh(
    query_points: torch.Tensor,
    body_vertices: torch.Tensor,
    body_faces: torch.Tensor,
) -> Binding:
    """Bind query points to their closest points on a triangle mesh."""
    query_points = _as_cpu_tensor(query_points, torch.float32)
    body_vertices = _as_cpu_tensor(body_vertices, torch.float32)
    body_faces = _as_cpu_tensor(body_faces, torch.int32)

    distances_squared, face_ids, closest_points = igl.point_mesh_squared_distance(
        query_points.numpy().astype(np.float64, copy=False),
        body_vertices.numpy().astype(np.float64, copy=False),
        body_faces.numpy(),
    )

    face_ids = torch.as_tensor(face_ids, dtype=torch.int32, device=torch.device("cpu"))
    closest_points = torch.as_tensor(closest_points, dtype=torch.float32, device=torch.device("cpu"))
    distances_squared = torch.as_tensor(
        distances_squared,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    triangles = body_vertices[body_faces.to(torch.long)[face_ids.to(torch.long)]]
    barycentric_coords = _barycentric_coordinates(closest_points, triangles)
    normals = _triangle_normals(triangles)

    return Binding(
        closest_points=closest_points,
        face_ids=face_ids,
        barycentric_coords=barycentric_coords,
        distances_squared=distances_squared,
        normals=normals,
    )


def _as_cpu_tensor(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().to(device=torch.device("cpu"), dtype=dtype)
    return torch.as_tensor(values, dtype=dtype, device=torch.device("cpu"))


def _barycentric_coordinates(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]

    v0 = b - a
    v1 = c - a
    v2 = points - a

    d00 = torch.sum(v0 * v0, dim=1)
    d01 = torch.sum(v0 * v1, dim=1)
    d11 = torch.sum(v1 * v1, dim=1)
    d20 = torch.sum(v2 * v0, dim=1)
    d21 = torch.sum(v2 * v1, dim=1)

    denom = d00 * d11 - d01 * d01
    assert bool(torch.all(torch.abs(denom) > 1e-12)), "Body mesh contains degenerate closest triangles."

    beta = (d11 * d20 - d01 * d21) / denom
    gamma = (d00 * d21 - d01 * d20) / denom
    alpha = 1.0 - beta - gamma
    return torch.stack([alpha, beta, gamma], dim=1)


def _triangle_normals(triangles: torch.Tensor) -> torch.Tensor:
    normals = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1)
    lengths = torch.linalg.norm(normals, dim=1, keepdim=True)
    assert bool(torch.all(lengths[:, 0] > 1e-12)), "Body mesh contains degenerate closest triangles."
    return normals / lengths
