from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import igl
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TestMeshPair:
    source_body_path: Path
    source_garment_path: Path
    target_body_path: Path
    body_scale_to_cm: float = 100.0


DEFAULT_TEST_PAIR = TestMeshPair(
    source_body_path=PROJECT_ROOT / "data/rand_2CU0AIB2VI/01258_apart.obj",
    source_garment_path=PROJECT_ROOT / "data/rand_2CU0AIB2VI/rand_2CU0AIB2VI_sim.ply",
    target_body_path=PROJECT_ROOT / "data/rand_0YN1FIW5GU/04737_apart.obj",
)


def load_triangle_mesh(path: Path, scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    vertices, faces = igl.read_triangle_mesh(str(path))
    vertices = torch.as_tensor(np.asarray(vertices * scale, dtype=np.float32), dtype=torch.float32)
    faces = torch.as_tensor(np.asarray(faces, dtype=np.int32), dtype=torch.int32)
    return vertices, faces


def load_default_test_pair() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    pair = DEFAULT_TEST_PAIR
    source_body_vertices, source_body_faces = load_triangle_mesh(
        pair.source_body_path,
        scale=pair.body_scale_to_cm,
    )
    source_garment_vertices, source_garment_faces = load_triangle_mesh(pair.source_garment_path)
    target_body_vertices, target_body_faces = load_triangle_mesh(
        pair.target_body_path,
        scale=pair.body_scale_to_cm,
    )
    return (
        source_body_vertices,
        source_body_faces,
        source_garment_vertices,
        source_garment_faces,
        target_body_vertices,
        target_body_faces,
    )
