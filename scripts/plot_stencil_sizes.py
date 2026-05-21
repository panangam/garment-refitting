from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from example_data import load_default_test_pair
from refitting.binding import closest_points_on_mesh
from refitting.stencil import construct_garment_stencils


def main() -> None:
    (
        source_body_vertices,
        source_body_faces,
        source_garment_vertices,
        source_garment_faces,
        _target_body_vertices,
        _target_body_faces,
    ) = load_default_test_pair()

    binding = closest_points_on_mesh(
        source_garment_vertices,
        source_body_vertices,
        source_body_faces,
    )
    stencils = construct_garment_stencils(source_garment_faces, binding.face_ids)
    stencil_sizes = torch.tensor([stencil.numel() for stencil in stencils], dtype=torch.int32)

    plt.figure()
    plt.hist(stencil_sizes.numpy(), bins=80)
    plt.xlabel("stencil size")
    plt.ylabel("garment vertex count")
    plt.title("Garment stencil sizes")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
