from __future__ import annotations

import matplotlib.pyplot as plt

from example_data import load_default_test_pair
from refitting.manager import GarmentRefittingManager


def main() -> None:
    (
        source_body_vertices,
        source_body_faces,
        source_garment_vertices,
        source_garment_faces,
        target_body_vertices,
        target_body_faces,
    ) = load_default_test_pair()

    manager = GarmentRefittingManager(
        source_garment_vertices,
        source_garment_faces,
        source_body_vertices,
        source_body_faces,
        target_body_vertices,
        target_body_faces,
    )
    history = manager.run_until_converged()

    plt.figure()
    plt.plot([entry.iteration_index for entry in history], [entry.max_movement for entry in history], label="max")
    plt.plot([entry.iteration_index for entry in history], [entry.mean_movement for entry in history], label="mean")
    plt.xlabel("iteration")
    plt.ylabel("movement (cm)")
    plt.title("Refitting movement history")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
