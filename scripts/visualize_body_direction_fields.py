from __future__ import annotations

import argparse
from pathlib import Path

import igl
import numpy as np
import polyscope as ps
import torch

from refitting.geometry_central import compute_face_tangent_basis, compute_smoothest_face_direction_field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BODY_SCALE_TO_CM = 100.0
DEFAULT_CURVE_RADIUS_CM = 0.03
DEFAULT_EDGE_WIDTH = 1.0
DEFAULT_MESH_GAP_CM = 10.0
DEFAULT_ARROW_LENGTH_FRACTION = 0.025


def load_obj_mesh(path: Path, scale: float = BODY_SCALE_TO_CM) -> tuple[torch.Tensor, torch.Tensor]:
    vertices, faces = igl.read_triangle_mesh(str(path))
    return (
        torch.as_tensor(np.asarray(vertices * scale, dtype=np.float32), dtype=torch.float32),
        torch.as_tensor(np.asarray(faces, dtype=np.int32), dtype=torch.int32),
    )


def face_centers(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return vertices[faces].mean(axis=1)


def face_direction_vectors(field: np.ndarray, face_tangent_basis: np.ndarray) -> np.ndarray:
    return field[:, 0, None] * face_tangent_basis[:, 0, :] + field[:, 1, None] * face_tangent_basis[:, 1, :]


def body_scaled_arrow_length(vertices: np.ndarray, arrow_length: float | None, arrow_length_fraction: float) -> float:
    if arrow_length is not None:
        return arrow_length
    bbox_low = vertices.min(axis=0)
    bbox_high = vertices.max(axis=0)
    return float((bbox_high - bbox_low).min() * arrow_length_fraction)


def direction_field_arrows(
    centers: np.ndarray,
    directions: np.ndarray,
    face_tangent_basis: np.ndarray,
    arrow_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    normals = np.cross(face_tangent_basis[:, 0, :], face_tangent_basis[:, 1, :])
    sides = np.cross(normals, directions)
    sides = sides / np.linalg.norm(sides, axis=1, keepdims=True)

    head_length = 0.28 * arrow_length
    head_width = 0.14 * arrow_length
    starts = centers
    tips = centers + arrow_length * directions
    head_bases = tips - head_length * directions
    left_heads = head_bases + head_width * sides
    right_heads = head_bases - head_width * sides

    nodes = np.empty((centers.shape[0] * 6, 3), dtype=np.float32)
    nodes[0::6] = starts
    nodes[1::6] = tips
    nodes[2::6] = tips
    nodes[3::6] = left_heads
    nodes[4::6] = tips
    nodes[5::6] = right_heads
    edges = np.arange(nodes.shape[0], dtype=np.int32).reshape(-1, 2)
    return nodes, edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--n-sym", type=int, default=1)
    parser.add_argument("--body-scale", type=float, default=BODY_SCALE_TO_CM)
    parser.add_argument("--arrow-length", type=float, default=None)
    parser.add_argument("--arrow-length-fraction", type=float, default=DEFAULT_ARROW_LENGTH_FRACTION)
    parser.add_argument("--curve-radius", type=float, default=DEFAULT_CURVE_RADIUS_CM)
    parser.add_argument("--edge-width", type=float, default=DEFAULT_EDGE_WIDTH)
    parser.add_argument("--mesh-gap", type=float, default=DEFAULT_MESH_GAP_CM)
    args = parser.parse_args()

    mesh_paths = sorted(args.data_dir.rglob("*.obj"))
    if not mesh_paths:
        raise FileNotFoundError(f"no OBJ body meshes found in {args.data_dir}")

    loaded_meshes: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    x_cursor = 0.0
    scene_points = []

    for path in mesh_paths:
        vertices, faces = load_obj_mesh(path, scale=args.body_scale)
        field = compute_smoothest_face_direction_field(vertices, faces, args.n_sym)
        face_tangent_basis = compute_face_tangent_basis(vertices, faces)

        vertices_np = vertices.numpy()
        faces_np = faces.numpy()
        field_np = field.numpy()
        face_tangent_basis_np = face_tangent_basis.numpy()

        width = vertices_np[:, 0].max() - vertices_np[:, 0].min()
        shifted_vertices = vertices_np + np.array([x_cursor - vertices_np[:, 0].min(), 0.0, 0.0], dtype=np.float32)
        x_cursor += width + max(args.mesh_gap, 0.25 * width)

        loaded_meshes.append((path, shifted_vertices, faces_np, field_np, face_tangent_basis_np))
        scene_points.append(shifted_vertices)

    all_points = np.vstack(scene_points)
    bbox_low = all_points.min(axis=0)
    bbox_high = all_points.max(axis=0)
    length_scale = np.linalg.norm(bbox_high - bbox_low)

    ps.set_autoscale_structures(False)
    ps.init()
    ps.set_automatically_compute_scene_extents(False)
    ps.set_bounding_box(bbox_low, bbox_high)
    ps.set_length_scale(length_scale)

    for path, vertices, faces, field, face_tangent_basis in loaded_meshes:
        name = path.relative_to(PROJECT_ROOT).as_posix()
        ps.register_surface_mesh(
            name,
            vertices,
            faces,
            edge_width=args.edge_width,
        )

        directions = face_direction_vectors(field, face_tangent_basis)
        centers = face_centers(vertices, faces)
        arrow_length = body_scaled_arrow_length(vertices, args.arrow_length, args.arrow_length_fraction)
        arrow_nodes, arrow_edges = direction_field_arrows(
            centers,
            directions,
            face_tangent_basis,
            arrow_length,
        )
        arrows = ps.register_curve_network(f"{name} direction field", arrow_nodes, arrow_edges)
        arrows.set_radius(args.curve_radius, relative=False)

    ps.show()


if __name__ == "__main__":
    main()
