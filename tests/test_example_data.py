import torch

from example_data import (
    DEFAULT_SOURCE_SET_ID,
    DEFAULT_TARGET_SET_ID,
    list_mesh_set_ids,
    load_default_test_pair,
    load_mesh_set,
    mesh_set_paths,
)


def test_default_test_pair_files_exist():
    """Checks the duplicated test fixture points at meshes present in the local data folder."""
    source_body_path, source_garment_path = mesh_set_paths(DEFAULT_SOURCE_SET_ID)
    target_body_path, target_garment_path = mesh_set_paths(DEFAULT_TARGET_SET_ID)

    assert source_body_path.exists()
    assert source_garment_path.exists()
    assert target_body_path.exists()
    assert target_garment_path.exists()


def test_load_mesh_set_loads_body_and_garment_tensors():
    """Checks the generalized set loader returns the body mesh and garment mesh tensors."""
    body_vertices, body_faces, garment_vertices, garment_faces = load_mesh_set(DEFAULT_SOURCE_SET_ID)

    assert body_vertices.dtype == torch.float32
    assert body_faces.dtype == torch.int32
    assert garment_vertices.dtype == torch.float32
    assert garment_faces.dtype == torch.int32
    assert body_vertices.shape[1] == 3
    assert body_faces.shape[1] == 3
    assert garment_vertices.shape[1] == 3
    assert garment_faces.shape[1] == 3


def test_list_mesh_set_ids_includes_default_sets():
    """Checks the dropdown set-id helper discovers the default source and target sets."""
    set_ids = list_mesh_set_ids()

    assert DEFAULT_SOURCE_SET_ID in set_ids
    assert DEFAULT_TARGET_SET_ID in set_ids


def test_default_test_pair_loads_expected_meshes():
    """Verifies the default source/target meshes load as CPU torch tensors with shared body faces."""
    (
        source_body_vertices,
        source_body_faces,
        source_garment_vertices,
        source_garment_faces,
        target_body_vertices,
        target_body_faces,
    ) = load_default_test_pair()

    assert source_body_vertices.dtype == torch.float32
    assert source_body_faces.dtype == torch.int32
    assert source_garment_vertices.dtype == torch.float32
    assert source_garment_faces.dtype == torch.int32
    assert target_body_vertices.dtype == torch.float32
    assert target_body_faces.dtype == torch.int32
    assert torch.equal(source_body_faces, target_body_faces)
    assert source_body_vertices.shape == target_body_vertices.shape
