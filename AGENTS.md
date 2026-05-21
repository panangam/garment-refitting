# AGENTS.md

## Environment

`uv` is used to manage the virtual environment, contained in `.venv` folder. Use uv to run python commands. 

## Implementation style

Implement the garment-refitting pipeline step by step, matching the implementation spec. Prefer simple, direct code over premature abstractions.

Use helper functions only when:
- the logic is used at least twice, or
- the function encapsulates a semantically meaningful part of the algorithm, such as closest-point binding, barycentric reconstruction, stencil construction, or relaxation solve.

Do not create one-off helper functions just to shorten a local block of code. If a local block of code seems long, add comment to break it up, but only sparingly.

Do not use fancy `__init__.py` files. Keep them empty, and use full path import. 

Do not add package dependencies by yourself. If a needed package is missing, tell me to manually add it.

## Assumptions

This code is for internal research use with mostly valid data. Do not over-engineer robustness. Add lightweight assertions or checks where they clarify assumptions, but avoid large validation frameworks, defensive wrappers, or excessive error handling.

Use:
- vertices as separate `(n, 3)` `float32` arrays,
- faces as separate `(m, 3)` `int32` arrays,
- Pytorch arrays throughout the implementation, both dense and sparse,
- libigl for closest-point queries,
- potpourri3d for parallel transport when that step is implemented.
- pytest for unit testing
- cholespy for sparse cholesky solving

## Development process

Implement one pipeline stage at a time and add small validation scripts or visualizations alongside it.

Use:
- matplotlib for 2D plots,
- Polyscope for 3D inspection.

Keep integration tests until the end, after the individual stages are working.

## References

The `references` folder contain reference file that can be checked if needed. The implementation spec should be complete, so only reach for these if necessary to not burn tokens. 