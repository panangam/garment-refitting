from __future__ import annotations

from pathlib import Path

import igl
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BODY_SCALE_TO_CM = 100.0
DEFAULT_SOURCE_SET_ID = "rand_0AAY3NQH8J"
DEFAULT_TARGET_SET_ID = "rand_1HX9UGCJ83"


def load_triangle_mesh(path: Path, scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    vertices, faces = igl.read_triangle_mesh(str(path))
    vertices = torch.as_tensor(np.asarray(vertices * scale, dtype=np.float32), dtype=torch.float32)
    faces = torch.as_tensor(np.asarray(faces, dtype=np.int32), dtype=torch.int32)
    return vertices, faces


def mesh_set_paths(set_id: str, data_root: Path = PROJECT_ROOT / "data") -> tuple[Path, Path]:
    set_dir = data_root / set_id
    body_paths = sorted(set_dir.glob("*_apart.obj"))
    assert len(body_paths) == 1, f"Expected exactly one body mesh matching '*_apart.obj' in {set_dir}."
    garment_path = set_dir / f"{set_id}_sim.ply"
    return body_paths[0], garment_path


def load_mesh_set(
    set_id: str,
    data_root: Path = PROJECT_ROOT / "data",
    body_scale_to_cm: float = BODY_SCALE_TO_CM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    body_path, garment_path = mesh_set_paths(set_id, data_root=data_root)
    body_vertices, body_faces = load_triangle_mesh(body_path, scale=body_scale_to_cm)
    garment_vertices, garment_faces = load_triangle_mesh(garment_path)
    return body_vertices, body_faces, garment_vertices, garment_faces


def load_default_test_pair() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    (
        source_body_vertices,
        source_body_faces,
        source_garment_vertices,
        source_garment_faces,
    ) = load_mesh_set(
        DEFAULT_SOURCE_SET_ID,
    )
    target_body_vertices, target_body_faces, _, _ = load_mesh_set(
        DEFAULT_TARGET_SET_ID,
    )
    return (
        source_body_vertices,
        source_body_faces,
        source_garment_vertices,
        source_garment_faces,
        target_body_vertices,
        target_body_faces,
    )
