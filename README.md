# Garment Refitting

This repository is a research implementation of garment refitting for body shape transfer, following de Goes, Fong, and O'Malley, ["Garment Refitting for Digital Characters"](https://research.pixar.com/#pub-2020-siggraphtalks-gfo) (SIGGRAPH Talks 2020). The pipeline alternates an affine-coordinate relaxation step with a rebinding step that preserves source garment spacing relative to a target body.

## Implementation

The implementation uses PyTorch tensors as the main in-memory data format for vertices, faces, sparse matrices, and intermediate geometry. Mesh closest-point queries and mass matrices use libigl, relaxation is solved with a prefactored sparse Cholesky system, and a narrow C++/nanobind binding to geometry-central provides the smooth face direction field and face tangent frames used by directional-field rebinding.

## Installation

Clone the repository with submodules, since the C++ geometry component is brought in that way:

```bash
git clone --recurse-submodules <repo-url>
cd garment-refitting
```

Then install the package locally:

```bash
pip install .
```

The project includes a C++ component based on `geometrycentral` for directional-field geometry computations; `pip install .` should build it automatically.

## Basic Usage

```python
from refitting.manager import GarmentRefittingManager

manager = GarmentRefittingManager(
    garment_vertices,  # torch.float32, (n_garment_vertices, 3)
    garment_faces,  # torch.int32, (n_garment_faces, 3)
    source_body_vertices,  # torch.float32, (n_body_vertices, 3)
    source_body_faces,  # torch.int32, (n_body_faces, 3)
    target_body_vertices,  # torch.float32, (n_body_vertices, 3)
    target_body_faces,  # torch.int32, (n_body_faces, 3)
)
refit_garment_vertices = manager.refit()
```

## Data Format

Example data, taken from GarmentCodeData, lives under `data/<sample_id>/`. Each sample includes a body mesh as an `*_apart.obj` file and a simulated source garment as an `*_sim.ply` file.

The current default test pair uses `data/rand_2CU0AIB2VI/01258_apart.obj` with `data/rand_2CU0AIB2VI/rand_2CU0AIB2VI_sim.ply` as the source, and `data/rand_0YN1FIW5GU/04737_apart.obj` as the target body. The body meshes share face connectivity. Body OBJs are loaded in meters and scaled by `100.0` into centimeters; garment PLYs are already treated as centimeters and are not rescaled.
