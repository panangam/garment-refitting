from __future__ import annotations

from dataclasses import dataclass

import cholespy
import torch

from refitting.affine_stencil import AffineStencilWeights


@dataclass(frozen=True)
class RelaxationSystem:
    matrix: torch.Tensor
    solver: cholespy.CholeskySolverF
    tightness_weights: torch.Tensor


def construct_stencil_matrix(
    center_vertex: int,
    stencil: torch.Tensor,
    num_vertices: int,
) -> torch.Tensor:
    num_edges = stencil.shape[0]
    rows = torch.arange(num_edges, dtype=torch.int64)
    indices = torch.cat(
        [
            torch.stack([rows, stencil.to(torch.int64)]),
            torch.stack([rows, torch.full((num_edges,), center_vertex, dtype=torch.int64)]),
        ],
        dim=1,
    )
    values = torch.cat(
        [
            torch.ones(num_edges, dtype=torch.float32),
            -torch.ones(num_edges, dtype=torch.float32),
        ]
    )
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_edges, num_vertices),
        check_invariants=False,
    ).coalesce()


def assemble_relaxation_system(
    num_vertices: int,
    affine_weights: AffineStencilWeights,
    tightness_weight: float = 1.0,
    vertex_areas: torch.Tensor | None = None,
    include_stencil_term: bool = True,
) -> RelaxationSystem:
    if vertex_areas is None:
        tightness_weights = torch.full((num_vertices,), tightness_weight, dtype=torch.float32)
    else:
        tightness_weights = tightness_weight * vertex_areas / torch.mean(vertex_areas)

    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []

    if include_stencil_term:
        for center, stencil, weight in zip(
            range(len(affine_weights.stencils)),
            affine_weights.stencils,
            affine_weights.weights,
            strict=True,
        ):
            stiffness = weight.T @ weight
            stencil_ids = stencil.tolist()
            for row_local, row_vertex in enumerate(stencil_ids):
                for col_local, col_vertex in enumerate(stencil_ids):
                    value = float(stiffness[row_local, col_local])
                    row_indices.append(row_vertex)
                    col_indices.append(col_vertex)
                    values.append(value)

                    row_indices.append(row_vertex)
                    col_indices.append(center)
                    values.append(-value)

                    row_indices.append(center)
                    col_indices.append(col_vertex)
                    values.append(-value)

                    row_indices.append(center)
                    col_indices.append(center)
                    values.append(value)

    for vertex_id in range(num_vertices):
        row_indices.append(vertex_id)
        col_indices.append(vertex_id)
        values.append(float(tightness_weights[vertex_id]))

    indices = torch.tensor([row_indices, col_indices], dtype=torch.int64)
    matrix = torch.sparse_coo_tensor(
        indices,
        torch.tensor(values, dtype=torch.float32),
        size=(num_vertices, num_vertices),
        check_invariants=False,
    ).coalesce()
    solver = _factorize_relaxation_matrix(matrix)
    return RelaxationSystem(matrix=matrix, solver=solver, tightness_weights=tightness_weights)


def solve_relaxation(
    system: RelaxationSystem,
    candidate_vertices: torch.Tensor,
) -> torch.Tensor:
    right_hand_side = system.tightness_weights[:, None] * candidate_vertices
    solution = torch.empty_like(right_hand_side)
    system.solver.solve(right_hand_side, solution)
    return solution


def _factorize_relaxation_matrix(matrix: torch.Tensor) -> cholespy.CholeskySolverF:
    matrix = matrix.coalesce()
    indices = matrix.indices()
    return cholespy.CholeskySolverF(
        matrix.shape[0],
        indices[0],
        indices[1],
        matrix.values(),
        cholespy.MatrixType.COO,
    )
