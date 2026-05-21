from __future__ import annotations

from dataclasses import dataclass

import torch

from refitting.affine_stencil import AffineStencilWeights, construct_affine_stencil_weights
from refitting.initial_warp import InitialWarp, compute_initial_warp
from refitting.rebinding import Rebinding, rebind_candidates_normal_aligned
from refitting.relaxation import RelaxationSystem, assemble_relaxation_system, solve_relaxation


@dataclass(frozen=True)
class IterationStats:
    iteration_index: int
    max_movement: float
    mean_movement: float
    closest_distance_min: float
    closest_distance_mean: float
    closest_distance_max: float
    converged: bool


class GarmentRefittingManager:
    def __init__(
        self,
        garment_vertices: torch.Tensor,
        garment_faces: torch.Tensor,
        source_body_vertices: torch.Tensor,
        source_body_faces: torch.Tensor,
        target_body_vertices: torch.Tensor,
        target_body_faces: torch.Tensor,
        tightness_weight: float = 1.0,
        max_iterations: int = 25,
        tolerance: float | None = None,
    ) -> None:
        self.garment_vertices = garment_vertices
        self.garment_faces = garment_faces
        self.source_body_vertices = source_body_vertices
        self.source_body_faces = source_body_faces
        self.target_body_vertices = target_body_vertices
        self.target_body_faces = target_body_faces
        self.tightness_weight = tightness_weight
        self.max_iterations = max_iterations

        self.initial_warp = compute_initial_warp(
            garment_vertices,
            source_body_vertices,
            source_body_faces,
            target_body_vertices,
            target_body_faces,
        )
        self.affine_weights = construct_affine_stencil_weights(
            garment_vertices,
            garment_faces,
            source_body_vertices,
            source_body_faces,
        )
        self.relaxation_system = assemble_relaxation_system(
            garment_vertices.shape[0],
            self.affine_weights,
            tightness_weight=tightness_weight,
        )

        self.current_candidate_vertices = self.initial_warp.candidate_vertices
        self.current_relaxed_vertices: torch.Tensor | None = None
        self.last_relaxation_input_vertices = self.current_candidate_vertices
        self.last_rebinding: Rebinding | None = None
        self.history: list[IterationStats] = []
        self.converged = False

        bbox_points = torch.cat([source_body_vertices, target_body_vertices, garment_vertices], dim=0)
        bbox_size = torch.linalg.norm(torch.max(bbox_points, dim=0).values - torch.min(bbox_points, dim=0).values)
        self.tolerance = float(0.0001 * bbox_size) if tolerance is None else tolerance

    def run_relaxation_step(self) -> torch.Tensor:
        self.last_relaxation_input_vertices = self.current_candidate_vertices
        self.current_relaxed_vertices = solve_relaxation(
            self.relaxation_system,
            self.current_candidate_vertices,
        )
        return self.current_relaxed_vertices

    def run_rebinding_step(self) -> Rebinding:
        if self.current_relaxed_vertices is None:
            self.run_relaxation_step()

        self.last_rebinding = rebind_candidates_normal_aligned(
            self.current_relaxed_vertices,
            self.target_body_vertices,
            self.target_body_faces,
            self.initial_warp.source_binding.closest_points,
            self.initial_warp.source_binding.normals,
            self.garment_vertices,
        )
        self.current_candidate_vertices = self.last_rebinding.candidate_vertices
        return self.last_rebinding

    def run_iteration(self) -> IterationStats:
        previous_vertices = self.current_candidate_vertices
        relaxed_vertices = self.run_relaxation_step()
        movement = torch.linalg.norm(relaxed_vertices - previous_vertices, dim=1)
        rebinding = self.run_rebinding_step()
        closest_distances = torch.sqrt(rebinding.target_binding.distances_squared)

        stats = IterationStats(
            iteration_index=len(self.history),
            max_movement=float(torch.max(movement)),
            mean_movement=float(torch.mean(movement)),
            closest_distance_min=float(torch.min(closest_distances)),
            closest_distance_mean=float(torch.mean(closest_distances)),
            closest_distance_max=float(torch.max(closest_distances)),
            converged=float(torch.max(movement)) < self.tolerance,
        )
        self.history.append(stats)
        self.converged = stats.converged or len(self.history) >= self.max_iterations
        return stats

    def run_until_converged(self) -> list[IterationStats]:
        while not self.converged:
            self.run_iteration()
        return self.history
