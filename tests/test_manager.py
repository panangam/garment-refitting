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


def _garment_and_body_meshes():
    garment_vertices = torch.tensor(
        [
            [0.0, 0.0, 0.2],
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.2],
            [0.0, 0.0, 1.2],
            [1.0, 1.0, 0.7],
        ],
        dtype=torch.float32,
    )
    garment_faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
            [1, 2, 3],
            [1, 2, 4],
        ],
        dtype=torch.int32,
    )
    body_vertices = garment_vertices.clone()
    body_vertices[:, 2] = 0.0
    body_faces = garment_faces.clone()
    return garment_vertices, garment_faces, body_vertices, body_faces
