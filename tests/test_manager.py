import igl
import numpy as np
import torch

from refitting.manager import GarmentRefittingManager


def test_manager_initializes_preprocessed_state_and_initial_candidate_vertices():
    """Checks the manager owns initial warp, affine weights, solver, and current Z state."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()

    manager = GarmentRefittingManager(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
        body_vertices,
        body_faces,
    )

    assert manager.current_candidate_vertices.shape == garment_vertices.shape
    assert manager.current_relaxed_vertices is None
    assert manager.relaxation_system.solver is manager.relaxation_system.solver
    assert len(manager.affine_weights.stencils) == garment_vertices.shape[0]
    assert manager.target_face_frame_field.frames.shape == (body_faces.shape[0], 3, 3)


def test_manager_can_run_relaxation_and_rebinding_steps_individually():
    """Verifies explicit one-step methods update relaxed X and rebound candidate Z state."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()
    manager = GarmentRefittingManager(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
        body_vertices,
        body_faces,
    )

    relaxed_vertices = manager.run_relaxation_step()
    rebinding = manager.run_rebinding_step()

    assert manager.current_relaxed_vertices is relaxed_vertices
    assert manager.last_rebinding is rebinding
    assert manager.current_candidate_vertices is rebinding.candidate_vertices
    assert rebinding.candidate_vertices.shape == garment_vertices.shape


def test_manager_can_run_directional_field_rebinding_step():
    """Verifies the manager can switch to the cached directional-field rebinding path."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()
    manager = GarmentRefittingManager(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
        body_vertices,
        body_faces,
        rebinding_method="directional_field",
    )

    relaxed_vertices = manager.run_relaxation_step()
    rebinding = manager.run_rebinding_step()

    assert manager.current_relaxed_vertices is relaxed_vertices
    assert manager.last_rebinding is rebinding
    assert rebinding.candidate_vertices.shape == garment_vertices.shape
    assert bool(torch.isfinite(rebinding.candidate_vertices).all())


def test_manager_can_run_normal_aligned_rebinding_step():
    """Verifies the manager can still use the normal-aligned fallback path."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()
    manager = GarmentRefittingManager(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
        body_vertices,
        body_faces,
        rebinding_method="normal_aligned",
    )

    manager.run_relaxation_step()
    rebinding = manager.run_rebinding_step()

    assert rebinding.candidate_vertices.shape == garment_vertices.shape
    assert bool(torch.isfinite(rebinding.candidate_vertices).all())


def test_manager_rebinding_is_idempotent_without_new_relaxation():
    """Checks repeated rebinding reuses the latest relaxed garment without advancing."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()
    target_body_vertices = body_vertices.clone()
    target_body_vertices[:, 0] += 0.25
    manager = GarmentRefittingManager(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
        target_body_vertices,
        body_faces,
    )

    manager.run_relaxation_step()
    first_rebinding = manager.run_rebinding_step()
    relaxed_vertices = manager.current_relaxed_vertices

    second_rebinding = manager.run_rebinding_step()

    assert manager.current_relaxed_vertices is relaxed_vertices
    torch.testing.assert_close(
        second_rebinding.candidate_vertices,
        first_rebinding.candidate_vertices,
        atol=1e-6,
        rtol=0.0,
    )


def test_manager_change_target_body_recomputes_initial_warp_and_keeps_solver():
    """Checks target-only swaps avoid rebuilding the garment relaxation system."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()
    manager = GarmentRefittingManager(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
        body_vertices,
        body_faces,
    )
    solver = manager.relaxation_system.solver
    old_candidate_vertices = manager.current_candidate_vertices
    manager.run_iteration()
    target_body_vertices = body_vertices.clone()
    target_body_vertices[:, 0] += 0.25

    manager.change_target_body(target_body_vertices, body_faces)

    assert manager.relaxation_system.solver is solver
    assert manager.current_relaxed_vertices is None
    assert manager.last_rebinding is None
    assert manager.history == []
    assert not manager.converged
    assert manager.target_body_vertices is target_body_vertices
    assert manager.target_face_frame_field.frames.shape == (body_faces.shape[0], 3, 3)
    assert not torch.allclose(manager.current_candidate_vertices, old_candidate_vertices)


def test_manager_change_tightness_weight_rebuilds_solver_and_keeps_geometry_preprocessing():
    """Checks tightness changes only rebuild the relaxation system and reset iteration state."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()
    manager = GarmentRefittingManager(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
        body_vertices,
        body_faces,
    )
    affine_weights = manager.affine_weights
    initial_warp = manager.initial_warp
    old_solver = manager.relaxation_system.solver
    manager.run_iteration()

    manager.change_tightness_weight(3.0)

    assert manager.tightness_weight == 3.0
    assert manager.affine_weights is affine_weights
    assert manager.initial_warp is initial_warp
    assert manager.relaxation_system.solver is not old_solver
    assert manager.current_relaxed_vertices is None
    assert manager.last_rebinding is None
    assert manager.history == []


def test_manager_uses_libigl_massmatrix_for_garment_vertex_areas():
    """Checks the tightness vertex areas come from libigl's mass matrix diagonal."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()
    manager = GarmentRefittingManager(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
        body_vertices,
        body_faces,
    )
    mass_matrix = igl.massmatrix(
        garment_vertices.numpy().astype(np.float64, copy=False),
        garment_faces.numpy().astype(np.int64, copy=False),
        igl.MASSMATRIX_TYPE_VORONOI,
    )

    torch.testing.assert_close(
        manager.garment_vertex_areas,
        torch.as_tensor(mass_matrix.diagonal(), dtype=torch.float32),
    )


def test_manager_run_iteration_records_finite_history():
    """Checks one alternating relaxation/rebinding iteration records movement and distance stats."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()
    manager = GarmentRefittingManager(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
        body_vertices,
        body_faces,
    )

    stats = manager.run_iteration()

    assert stats in manager.history
    assert stats.iteration_index == 0
    assert stats.max_movement >= 0.0
    assert stats.mean_movement >= 0.0
    assert stats.closest_distance_min >= 0.0
    assert stats.closest_distance_max >= stats.closest_distance_min


def test_manager_identical_source_target_converges_quickly():
    """Verifies the identical source/target case converges in the first manager iteration."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()
    manager = GarmentRefittingManager(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
        body_vertices,
        body_faces,
    )

    stats = manager.run_iteration()

    assert stats.converged
    assert manager.converged


def test_manager_run_until_converged_respects_max_iterations():
    """Checks the outer loop stops no later than the configured iteration cap."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()
    target_body_vertices = body_vertices.clone()
    target_body_vertices[:, 0] += 0.25
    manager = GarmentRefittingManager(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
        target_body_vertices,
        body_faces,
        max_iterations=2,
        tolerance=0.0,
    )

    history = manager.run_until_converged()

    assert len(history) <= 2
    assert manager.converged


def test_manager_refit_returns_last_relaxed_garment_and_keeps_history():
    """Checks the simple entry point returns final garment vertices while preserving stats."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()
    target_body_vertices = body_vertices.clone()
    target_body_vertices[:, 0] += 0.25
    manager = GarmentRefittingManager(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
        target_body_vertices,
        body_faces,
        max_iterations=2,
        tolerance=0.0,
    )

    refit_vertices = manager.refit()

    assert refit_vertices is manager.current_relaxed_vertices
    assert manager.converged
    assert len(manager.history) <= 2


def test_manager_reset_restores_initial_warp_state_and_keeps_preprocessing():
    """Checks reset clears iteration state while preserving cached preprocessing and solver."""
    garment_vertices, garment_faces, body_vertices, body_faces = _garment_and_body_meshes()
    manager = GarmentRefittingManager(
        garment_vertices,
        garment_faces,
        body_vertices,
        body_faces,
        body_vertices,
        body_faces,
    )
    solver = manager.relaxation_system.solver
    manager.run_iteration()

    manager.reset()

    assert manager.relaxation_system.solver is solver
    assert manager.current_relaxed_vertices is None
    assert manager.last_rebinding is None
    assert manager.history == []
    assert not manager.converged
    assert manager.current_candidate_vertices is manager.initial_warp.candidate_vertices


def _garment_and_body_meshes():
    body_vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    body_faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=torch.int32,
    )
    garment_vertices = body_vertices + torch.tensor([0.0, 0.0, 0.2], dtype=torch.float32)
    garment_faces = body_faces.clone()
    return garment_vertices, garment_faces, body_vertices, body_faces
