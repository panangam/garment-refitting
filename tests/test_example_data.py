import torch

from example_data import DEFAULT_TEST_PAIR, load_default_test_pair


def test_default_test_pair_files_exist():
    assert DEFAULT_TEST_PAIR.source_body_path.exists()
    assert DEFAULT_TEST_PAIR.source_garment_path.exists()
    assert DEFAULT_TEST_PAIR.target_body_path.exists()


def test_default_test_pair_loads_expected_meshes():
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
