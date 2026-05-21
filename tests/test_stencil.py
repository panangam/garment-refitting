import torch

from refitting.stencil import construct_garment_stencils


def test_construct_garment_stencils_includes_all_face_adjacent_vertices_on_nonmanifold_mesh():
    garment_faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 4, 5],
        ],
        dtype=torch.int32,
    )
    closest_face_ids = torch.arange(6, dtype=torch.int32)

    stencils = construct_garment_stencils(garment_faces, closest_face_ids)

    assert _as_set(stencils[0]) == {1, 2, 3, 4, 5}
    assert {0, 2}.issubset(_as_set(stencils[1]))
    assert {0, 1, 3}.issubset(_as_set(stencils[2]))
    assert {0, 2}.issubset(_as_set(stencils[3]))
    assert {0, 5}.issubset(_as_set(stencils[4]))
    assert {0, 4}.issubset(_as_set(stencils[5]))


def test_construct_garment_stencils_includes_vertices_sharing_closest_source_face():
    garment_faces = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=torch.int32,
    )
    closest_face_ids = torch.tensor([10, 20, 10, 20, 10, 30], dtype=torch.int32)

    stencils = construct_garment_stencils(garment_faces, closest_face_ids)

    assert {2, 4}.issubset(_as_set(stencils[0]))
    assert {0, 4}.issubset(_as_set(stencils[2]))
    assert {0, 2}.issubset(_as_set(stencils[4]))
    assert {1, 4}.issubset(_as_set(stencils[3]))


def test_construct_garment_stencils_removes_center_and_uses_deterministic_order():
    garment_faces = torch.tensor(
        [
            [2, 0, 1],
            [2, 4, 3],
        ],
        dtype=torch.int32,
    )
    closest_face_ids = torch.tensor([7, 7, 8, 7, 8], dtype=torch.int32)

    stencils = construct_garment_stencils(garment_faces, closest_face_ids)

    for center, stencil in enumerate(stencils):
        assert center not in stencil.tolist()
        assert stencil.tolist() == sorted(stencil.tolist())

    assert stencils[2].tolist() == [0, 1, 3, 4]


def _as_set(stencil: torch.Tensor) -> set[int]:
    return set(stencil.tolist())
