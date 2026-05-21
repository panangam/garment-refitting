# Garment Refitting

## Data Format

Example data lives under `data/<sample_id>/`. Each sample includes a body mesh as an `*_apart.obj` file and a simulated source garment as an `*_sim.ply` file.

The current default test pair uses `data/rand_2CU0AIB2VI/01258_apart.obj` with `data/rand_2CU0AIB2VI/rand_2CU0AIB2VI_sim.ply` as the source, and `data/rand_0YN1FIW5GU/04737_apart.obj` as the target body. The body meshes share face connectivity. Body OBJs are loaded in meters and scaled by `100.0` into centimeters; garment PLYs are already treated as centimeters and are not rescaled.
