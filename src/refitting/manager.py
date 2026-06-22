"""Documented entry point for the garment refitting pipeline.

The main API is :class:`GarmentRefittingManager`. It owns the reusable
preprocessing for one source garment/body pair and one target body:

- initial closest-point warp from source body to target body,
- target-body smooth face direction frames for directional-field rebinding,
- garment affine stencil weights and the sparse relaxation system,
- per-iteration relaxed vertices, rebound candidate vertices, and convergence
  statistics.

Construct the manager with CPU torch tensors:

- ``garment_vertices`` and body vertices as ``(n, 3)`` ``float32`` tensors,
- ``garment_faces`` and body faces as ``(m, 3)`` ``int32`` tensors.

The default rebinding method is ``"directional_field"``, which transports each
source displacement through the target body's cached face frame field. Use
``run_relaxation_step()`` and ``run_rebinding_step()`` for manual inspection, or
``run_until_converged()`` for the full alternating solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import igl
import numpy as np
import torch

from refitting.affine_stencil import AffineStencilWeights, construct_affine_stencil_weights
from refitting.directional_field import compute_face_frame_field
from refitting.initial_warp import InitialWarp, compute_initial_warp
from refitting.rebinding import Rebinding, rebind_candidates_directional_field, rebind_candidates_normal_aligned
from refitting.relaxation import (
    RelaxationSystem,
    assemble_relaxation_system,
    solve_relaxation,
)


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
    """Manage preprocessing, iteration state, and solves for garment refitting.

    The manager is intentionally stateful: each relaxation step consumes the
    latest rebound candidate vertices, and each rebinding step consumes the
    latest relaxed vertices. ``reset()`` returns the garment to the initial warp.
    ``change_target_body()`` keeps garment-side preprocessing but recomputes the
    target-dependent warp and direction frames.
    """

    def __init__(
        self,
        garment_vertices: torch.Tensor,
        garment_faces: torch.Tensor,
        source_body_vertices: torch.Tensor,
        source_body_faces: torch.Tensor,
        target_body_vertices: torch.Tensor,
        target_body_faces: torch.Tensor,
        tightness_weight: float = 0.1,
        max_iterations: int = 25,
        tolerance: float | None = None,
        rebinding_method: str = "directional_field",
    ) -> None:
        """Precompute reusable state for refitting one garment to one target body.

        Args:
            garment_vertices: Source garment vertices as an ``(n, 3)`` float32
                tensor.
            garment_faces: Source garment triangle faces as an ``(m, 3)`` int32
                tensor.
            source_body_vertices: Source body vertices as an ``(n, 3)`` float32
                tensor.
            source_body_faces: Source body triangle faces as an ``(m, 3)`` int32
                tensor.
            target_body_vertices: Target body vertices as an ``(n, 3)`` float32
                tensor.
            target_body_faces: Target body triangle faces as an ``(m, 3)`` int32
                tensor.
            tightness_weight: Weight for the garment-to-candidate positional
                term in the relaxation solve.
            max_iterations: Maximum number of relaxation/rebinding iterations
                used by ``run_until_converged()``.
            tolerance: Convergence tolerance for maximum vertex movement. If
                ``None``, a scale-relative tolerance is computed.
            rebinding_method: Rebinding method name, either
                ``"directional_field"`` or ``"normal_aligned"``.

        Returns:
            None.
        """

        assert rebinding_method in {
            "normal_aligned",
            "directional_field",
        }, "rebinding_method must be 'normal_aligned' or 'directional_field'."
        self.garment_vertices = garment_vertices
        self.garment_faces = garment_faces
        self.source_body_vertices = source_body_vertices
        self.source_body_faces = source_body_faces
        self.target_body_vertices = target_body_vertices
        self.target_body_faces = target_body_faces
        self.tightness_weight = tightness_weight
        self.max_iterations = max_iterations
        self.rebinding_method = rebinding_method
        self._automatic_tolerance = tolerance is None

        self.initial_warp = compute_initial_warp(
            garment_vertices,
            source_body_vertices,
            source_body_faces,
            target_body_vertices,
            target_body_faces,
        )
        self.target_face_frame_field = compute_face_frame_field(
            target_body_vertices,
            target_body_faces,
            n_sym=1,
        )
        self.affine_weights = construct_affine_stencil_weights(
            garment_vertices,
            garment_faces,
            source_body_vertices,
            source_body_faces,
        )
        garment_mass_matrix = igl.massmatrix(
            garment_vertices.numpy().astype(np.float64, copy=False),
            garment_faces.numpy().astype(np.int64, copy=False),
            igl.MASSMATRIX_TYPE_VORONOI,
        )
        self.garment_vertex_areas = torch.as_tensor(
            garment_mass_matrix.diagonal(),
            dtype=torch.float32,
        )
        self.relaxation_system = assemble_relaxation_system(
            garment_vertices.shape[0],
            self.affine_weights,
            tightness_weight=tightness_weight,
            vertex_areas=self.garment_vertex_areas,
        )

        self.current_candidate_vertices = self.initial_warp.candidate_vertices
        self.current_relaxed_vertices: torch.Tensor | None = None
        self.last_rebinding: Rebinding | None = None
        self.history: list[IterationStats] = []
        self.converged = False

        self.tolerance = 0.0
        self._update_tolerance(tolerance)
        self.reset()

    def change_target_body(
        self,
        target_body_vertices: torch.Tensor,
        target_body_faces: torch.Tensor,
    ) -> None:
        """Switch to a new target body while keeping garment-side preprocessing.

        This recomputes the initial warp and target face frames, then resets the
        current relaxed/rebound state to the new initial warp.

        Args:
            target_body_vertices: New target body vertices as an ``(n, 3)``
                float32 tensor.
            target_body_faces: New target body triangle faces as an ``(m, 3)``
                int32 tensor.

        Returns:
            None.
        """

        self.target_body_vertices = target_body_vertices
        self.target_body_faces = target_body_faces
        self.initial_warp = compute_initial_warp(
            self.garment_vertices,
            self.source_body_vertices,
            self.source_body_faces,
            target_body_vertices,
            target_body_faces,
        )
        self.target_face_frame_field = compute_face_frame_field(
            target_body_vertices,
            target_body_faces,
            n_sym=1,
        )
        if self._automatic_tolerance:
            self._update_tolerance(None)
        self.reset()

    def change_tightness_weight(self, tightness_weight: float) -> None:
        """Rebuild the relaxation system with a new tightness weight and reset.

        Args:
            tightness_weight: New weight for the garment-to-candidate positional
                term in the relaxation solve.

        Returns:
            None.
        """

        self.tightness_weight = tightness_weight
        self.relaxation_system = assemble_relaxation_system(
            self.garment_vertices.shape[0],
            self.affine_weights,
            tightness_weight=tightness_weight,
            vertex_areas=self.garment_vertex_areas,
        )
        self.reset()

    def _update_tolerance(self, tolerance: float | None) -> None:
        """Set convergence tolerance.

        Args:
            tolerance: Explicit maximum-movement convergence tolerance. If
                ``None``, use a bounding-box-relative value.

        Returns:
            None.
        """

        bbox_points = torch.cat(
            [self.source_body_vertices, self.target_body_vertices, self.garment_vertices],
            dim=0,
        )
        bbox_size = torch.linalg.norm(torch.max(bbox_points, dim=0).values - torch.min(bbox_points, dim=0).values)
        self.tolerance = float(0.0001 * bbox_size) if tolerance is None else tolerance

    def reset(self) -> None:
        """Return iteration state to the current initial warp.

        Returns:
            None.
        """

        self.current_candidate_vertices = self.initial_warp.candidate_vertices
        self.current_relaxed_vertices = None
        self.last_rebinding = None
        self.history.clear()
        self.converged = False

    def run_relaxation_step(self) -> torch.Tensor:
        """Solve one relaxation step from the latest rebound candidate vertices.

        Returns:
            Relaxed garment vertices as an ``(n, 3)`` float32 tensor.
        """

        self.current_relaxed_vertices = solve_relaxation(
            self.relaxation_system,
            self.current_candidate_vertices,
        )
        return self.current_relaxed_vertices

    def run_rebinding_step(self) -> Rebinding:
        """Project the latest relaxed garment back to the target body.

        If no relaxation step has been run since reset, this rebinds the current
        candidate vertices directly. The returned ``Rebinding`` also becomes
        ``last_rebinding`` and its candidate vertices become the next iteration's
        input.

        Returns:
            Rebinding result containing closest-point target binding data and
            the next candidate garment vertices.
        """

        relaxed_vertices = (
            self.current_candidate_vertices
            if self.current_relaxed_vertices is None
            else self.current_relaxed_vertices
        )

        if self.rebinding_method == "normal_aligned":
            self.last_rebinding = rebind_candidates_normal_aligned(
                relaxed_vertices,
                self.target_body_vertices,
                self.target_body_faces,
                self.initial_warp.source_binding.closest_points,
                self.initial_warp.source_binding.normals,
                self.garment_vertices,
            )
        else:
            self.last_rebinding = rebind_candidates_directional_field(
                relaxed_vertices,
                self.target_body_vertices,
                self.target_body_faces,
                self.initial_warp.source_binding.face_ids,
                self.initial_warp.reoriented_displacements,
                self.target_face_frame_field,
            )
        self.current_candidate_vertices = self.last_rebinding.candidate_vertices
        return self.last_rebinding

    def run_iteration(self) -> IterationStats:
        """Run one relaxation/rebinding iteration and append convergence stats.

        Returns:
            Statistics for the completed iteration.
        """

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
        """Run iterations until convergence or ``max_iterations`` is reached.

        Returns:
            The accumulated per-iteration statistics.
        """

        while not self.converged:
            self.run_iteration()
        return self.history

    def refit(self) -> torch.Tensor:
        """Run the standard solve and return the final refit garment vertices.

        This is the simple entry point for the common use case where only the
        last relaxed garment is needed. Per-iteration convergence statistics
        remain available afterward through ``manager.history``; use
        ``run_until_converged()`` directly when you want the history as the
        return value.

        Returns:
            Final relaxed garment vertices as an ``(n, 3)`` float32 tensor.
        """

        self.run_until_converged()
        assert self.current_relaxed_vertices is not None
        return self.current_relaxed_vertices
