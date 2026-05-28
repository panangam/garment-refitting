import torch

from refitting.affine_stencil import AffineStencilWeights
from refitting.affine_stencil import construct_affine_stencil_weights
from refitting.relaxation import (
    assemble_relaxation_system,
    construct_stencil_matrix,
    solve_relaxation,
)


def test_construct_stencil_matrix_extracts_edge_vectors():
    """Checks B_v X equals the ordered stencil edge vectors x_i - x_v."""
    stencil = torch.tensor([0, 2, 4], dtype=torch.int32)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [1.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    stencil_matrix = construct_stencil_matrix(center_vertex=1, stencil=stencil, num_vertices=5)
    edges = torch.sparse.mm(stencil_matrix, positions)

    torch.testing.assert_close(edges, positions[stencil.to(torch.long)] - positions[1])
    assert stencil_matrix.shape == (3, 5)
    assert stencil_matrix._nnz() == 6


def test_assemble_relaxation_system_shape_symmetry_and_positive_definite_quadratic():
    """Checks H has the expected sparse shape, symmetry, and positive quadratic form."""
    affine_weights = _affine_weights_for_tests()

    system = assemble_relaxation_system(4, affine_weights, tightness_weight=0.25)
    dense_matrix = system.matrix.to_dense()

    assert system.matrix.shape == (4, 4)
    torch.testing.assert_close(dense_matrix, dense_matrix.T, atol=1e-6, rtol=0.0)
    for seed in range(3):
        torch.manual_seed(seed)
        y = torch.randn(4, dtype=torch.float32)
        assert float(y @ dense_matrix @ y) > 0.0


def test_assemble_relaxation_system_uses_vertex_area_tightness_weights():
    """Verifies M uses area-weighted tightness on both the matrix and RHS."""
    affine_weights = AffineStencilWeights(stencils=[], weights=[])
    vertex_areas = torch.tensor([0.5, 2.0, 3.0], dtype=torch.float32)
    candidate_vertices = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=torch.float32,
    )

    system = assemble_relaxation_system(
        3,
        affine_weights,
        tightness_weight=4.0,
        vertex_areas=vertex_areas,
        include_stencil_term=False,
    )
    relaxed = solve_relaxation(system, candidate_vertices)

    torch.testing.assert_close(system.tightness_weights, 4.0 * vertex_areas)
    torch.testing.assert_close(
        system.matrix.to_dense(),
        torch.diag(4.0 * vertex_areas),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(relaxed, candidate_vertices, atol=1e-6, rtol=0.0)


def test_solve_relaxation_with_stencil_term_disabled_returns_candidates():
    """Verifies the implicit tightness-only system solves exactly to Z."""
    affine_weights = AffineStencilWeights(stencils=[], weights=[])
    candidate_vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    system = assemble_relaxation_system(
        3,
        affine_weights,
        tightness_weight=2.0,
        include_stencil_term=False,
    )
    relaxed = solve_relaxation(system, candidate_vertices)

    torch.testing.assert_close(relaxed, candidate_vertices, atol=1e-6, rtol=0.0)
    assert system.solver is system.solver


def test_solve_relaxation_reuses_factorized_solver_for_multiple_rhs():
    """Checks a single relaxation system can solve multiple candidate vertex matrices."""
    affine_weights = _affine_weights_for_tests()
    system = assemble_relaxation_system(4, affine_weights, tightness_weight=1.0)
    first_candidates = torch.zeros((4, 3), dtype=torch.float32)
    second_candidates = torch.ones((4, 3), dtype=torch.float32)
    solver = system.solver

    first_relaxed = solve_relaxation(system, first_candidates)
    second_relaxed = solve_relaxation(system, second_candidates)

    assert system.solver is solver
    torch.testing.assert_close(first_relaxed, torch.zeros_like(first_relaxed), atol=1e-6, rtol=0.0)
    assert bool(torch.isfinite(second_relaxed).all())


def test_solve_relaxation_output_is_finite_and_tightness_pulls_toward_candidates():
    """Checks solve output is finite and larger tightness stays closer to the candidates."""
    affine_weights = _affine_weights_for_tests()
    candidate_vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    loose = solve_relaxation(
        assemble_relaxation_system(4, affine_weights, tightness_weight=0.1),
        candidate_vertices,
    )
    tight = solve_relaxation(
        assemble_relaxation_system(4, affine_weights, tightness_weight=10.0),
        candidate_vertices,
    )

    assert loose.shape == candidate_vertices.shape
    assert bool(torch.isfinite(loose).all())
    assert torch.linalg.norm(tight - candidate_vertices) < torch.linalg.norm(loose - candidate_vertices)


def test_solve_relaxation_keeps_identical_source_target_garment_close():
    """Checks relaxation leaves the original garment nearly fixed when Z is the original garment."""
    garment_vertices, garment_faces = _garment_and_body_meshes()
    affine_weights = construct_affine_stencil_weights(
        garment_vertices,
        garment_faces,
        garment_vertices,
        garment_faces,
    )
    system = assemble_relaxation_system(
        garment_vertices.shape[0],
        affine_weights,
        tightness_weight=1.0,
    )

    relaxed = solve_relaxation(system, garment_vertices)

    torch.testing.assert_close(relaxed, garment_vertices, atol=1e-5, rtol=0.0)


def _affine_weights_for_tests() -> AffineStencilWeights:
    stencils = [
        torch.tensor([1, 2, 3], dtype=torch.int32),
        torch.tensor([0, 2, 3], dtype=torch.int32),
        torch.tensor([0, 1, 3], dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
    ]
    weights = [
        torch.eye(stencil.shape[0], dtype=torch.float32)
        for stencil in stencils
    ]
    return AffineStencilWeights(stencils=stencils, weights=weights)


def _garment_and_body_meshes():
    garment_vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.5],
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
    return garment_vertices, garment_faces
