from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from example_data import load_default_test_pair
from refitting.affine_stencil import construct_affine_stencil_weights


def main() -> None:
    (
        source_body_vertices,
        source_body_faces,
        source_garment_vertices,
        source_garment_faces,
        _target_body_vertices,
        _target_body_faces,
    ) = load_default_test_pair()

    affine_weights = construct_affine_stencil_weights(
        source_garment_vertices,
        source_garment_faces,
        source_body_vertices,
        source_body_faces,
    )

    residual_norms = []
    for center, stencil, weight in zip(
        range(len(affine_weights.stencils)),
        affine_weights.stencils,
        affine_weights.weights,
        strict=True,
    ):
        edges = source_garment_vertices[stencil.to(torch.long)] - source_garment_vertices[center]
        residual_norms.append(torch.linalg.norm(edges.T @ weight))

    residual_norms = torch.stack(residual_norms).numpy()

    plt.figure()
    plt.semilogy(residual_norms)
    plt.xlabel("garment vertex")
    plt.ylabel("original stencil residual norm")
    plt.title("Affine-invariant stencil residuals")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
