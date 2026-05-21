# Garment Refitting Implementation Spec

This document is intended to guide a Codex implementation. The goal is to implement the garment-refitting pipeline in small pieces that can be validated one at a time.

Assumptions for this implementation:

- Vertices are stored as separate dense matrices of shape `(n, 3)` using `float32`.
- Faces are stored as separate dense matrices of shape `(m, 3)` using `int32`.
- Source and target body meshes share the same face connectivity.
- Use pytorch dense and sparse tensor throughout the implementation, but the geometric processing and sparse linear algebra can use SciPy, libigl, and potpourri3d internally. Use only CPU tensors for now. 
- Skip cloth-simulation post-processing.

## 1. Pipeline overview

The input is an original garment mesh already fitted to a source body `S`, and the goal is to produce a refitted garment for a target body `T`. The source and target body meshes share connectivity, so a point bound to a face on `S` can be reconstructed on the corresponding face of `T` using the same face index and barycentric coordinates.

The pipeline has four stages.

### Pre-processing

First, bind every original garment vertex to the source body by finding its closest point on the source body. For each garment vertex `v`, store the closest source-body point, closest source-body face, barycentric coordinates on that face, source normal, and displacement vector from the body point to the garment vertex.

Use `igl.point_mesh_squared_distance` for closest-point queries. Compute barycentric coordinates from the returned closest point and triangle.

Use uniform positive tightness weights for the first version.

Build a stencil `N_v` for each garment vertex using two vertex sets:

1. one-ring garment neighbors of `v`, without assuming manifoldness,
2. multi-layer / fold-over augmentation: all garment vertices whose closest source-body point lies on the same closest source-body face.

Skip UV panel-boundary augmentation for the first version.

For each stencil, construct an affine-invariant matrix `W_v` from the original garment positions. Let the original stencil edge matrix be:

$$
\bar E_v = [\ldots,\ \bar x_i - \bar x_v,\ \ldots]
$$

where `i` ranges over the ordered vertices in `N_v`. Define:

$$
C_v = (\bar E_v \bar E_v^T)^\dagger
$$

$$
W_v = I_v - \bar E_v^T C_v \bar E_v
$$

Here `†` denotes the pseudoinverse and `I_v` is the identity matrix of size `n_v × n_v`. Interpret `W_v` as extracting the part of the stencil deformation that cannot be explained by a local affine transformation.

### Initial warp

Use the source-body binding to produce initial candidate garment vertices on the target body.

For each garment vertex, reconstruct the corresponding body point on `T` using the stored source face ID and barycentric coordinates. Then reorient the original displacement vector from the source normal orientation to the target normal orientation, and add it to the reconstructed target-body point to get the initial candidate vertex `z_v`.

Do not use surface parallel transport during initial warp, because this step maps between two different geometries rather than transporting between two points on one surface.

### Optimization

Iteratively alternate relaxation and rebinding.

In relaxation, solve for refitted garment vertices `X` that preserve the original garment’s local affine structure while staying close to the current candidate vertices `Z`.

The relaxation energy is:

$$
E(X) = \frac{1}{2} \sum_v \|W_v B_v X\|_F^2 + \frac{1}{2} \|X - Z\|_M^2
$$

where:

- `X` is the unknown garment vertex matrix of shape `(n_g, 3)`,
- `Z` is the current candidate vertex matrix of shape `(n_g, 3)`,
- `B_v` is the sparse stencil edge-extraction matrix for vertex `v`,
- `M = diag(m_v)` is the diagonal tightness-weight matrix,
- `||·||_F` is the Frobenius norm, meaning the square root of the sum of squared matrix entries,
- `||X - Z||_M^2 = tr((X - Z)^T M (X - Z))`, where `tr` means matrix trace.

Each row of `B_v` corresponds to one stencil vertex `i ∈ N_v`: it has `+1` at stencil vertex `i`, `-1` at center vertex `v`, and zeros elsewhere. Thus `B_v X` gives the current stencil edge vectors `x_i - x_v`.

The relaxation linear system is:

$$
H X = R
$$

with:

$$
H = \sum_v B_v^T W_v^T W_v B_v + M
$$

and:

$$
R = MZ
$$

`H` is symmetric positive semidefinite because it is a sum of squared linear residual terms. With uniform positive tightness weights, `M` is positive definite, so `H` is positive definite and can be prefactored.

In rebinding, find new closest points from the relaxed garment vertices to the target body, reorient/transport the stored displacement vectors to those new closest points, and update the candidate vertices `Z`.

For rebinding displacement transport, use direct parallel transport rather than constructing the paper’s global frame field. The preferred option is `potpourri3d.transport_tangent_vector`. Since it transports tangent vectors, decompose each displacement into tangent and normal components. Transport the tangent component on the target body, carry the normal component using the new target-body normal, and recombine. As a fallback, use naive normal-aligned rotation.

Terminate the optimization when the largest garment vertex movement is below `0.01%` of the model bounding-box size, or when `25` iterations have been reached.

### Post-processing

The paper uses cloth simulation after refitting to resolve garment/body intersections and add draping detail. Skip that step in this implementation.

## 2. Steps to implement and validate

Implement the following pieces in order. Add debug visualizations along the way when they help verify the module. Use matplotlib for 2D plots and Polyscope for 3D visualization.

### Step 1: Closest-point binding helper

Implement a reusable helper:

```python
closest_points_on_mesh(query_points, body_vertices, body_faces) -> Binding
```

`Binding` should contain:

- `closest_points`: `(n_query, 3) float32`,
- `face_ids`: `(n_query,) int32`,
- `barycentric_coords`: `(n_query, 3) float32`,
- `distances_squared`: `(n_query,) float32`,
- `normals`: `(n_query, 3) float32`.

Use `igl.point_mesh_squared_distance` to get closest points, squared distances, and face IDs. Then compute barycentric coordinates on the returned closest triangle.

Validation:

- Reconstruct the closest points from `face_ids` and `barycentric_coords`; compare to the returned closest points.
- Check barycentric weights sum to `1` within tolerance.
- Check all face IDs are valid.
- Check all normals are finite and unit length within tolerance.
- Polyscope visualization: show the body mesh, query points, closest points, and short line segments connecting each query point to its closest point.

### Step 2: Barycentric reconstruction helper

Implement:

```python
reconstruct_from_barycentric(body_vertices, body_faces, face_ids, barycentric_coords) -> points
```

This should reconstruct points using the body triangle vertices selected by `face_ids` and the corresponding barycentric coordinates.

Validation:

- Applying this helper to the output of Step 1 should reproduce `Binding.closest_points`.
- On a single known triangle, test corners, edge midpoints, and the triangle centroid.
- Polyscope visualization: display reconstructed points on top of the mesh to confirm they lie on the intended faces.

### Step 3: Initial source binding

Using Step 1, bind original garment vertices to the source body.

For each garment vertex `v`, store:

- source closest point `p_v^S`,
- source face ID `f_v^S`,
- barycentric coordinates on `f_v^S`,
- source normal at the closest point,
- original displacement `d_v = x_v - p_v^S`.

Validation:

- Check `x_v = p_v^S + d_v` within tolerance.
- Plot a histogram of closest-point distances using matplotlib.
- Polyscope visualization: show source body, garment, closest points, and displacement vectors.

### Step 4: Uniform tightness weights

Implement uniform positive tightness weights.

Output:

- vector `m` of shape `(n_g,)`,
- sparse diagonal matrix `M = diag(m_v)`.

Validation:

- All weights are positive.
- `M` is diagonal with the expected entries.
- Changing the scalar tightness value changes `M` but not any stencil or geometry data.

### Step 5: Garment stencil construction

Construct ordered stencils `N_v` for each garment vertex.

Include:

1. one-ring garment neighbors from garment faces,
2. all garment vertices whose closest source-body face ID equals `f_v^S`.

Implementation choices:

- Deduplicate vertices.
- Remove the center vertex `v` from its own stencil.
- Do not assume garment manifoldness.
- Preserve a deterministic order for each stencil, since `W_v` must match the stencil order.
- Ignore the edge/vertex closest-point ambiguity for the first version: each garment vertex belongs to exactly one closest-face cluster.

Validation:

- On a small synthetic nonmanifold mesh, check that all face-adjacent vertices are included.
- Check that vertices sharing the same closest source-body face are included in each other’s stencils.
- Check no stencil contains its own center vertex.
- Plot a histogram of stencil sizes using matplotlib.
- Polyscope visualization: select a few vertices and show their stencil vertices as highlighted point clouds or small spheres.

### Step 6: Affine-invariant `W_v` construction

For each garment vertex, use the original garment positions and the ordered stencil `N_v` to construct `W_v`.

For each `v`, build:

$$
\bar E_v = [\ldots,\ \bar x_i - \bar x_v,\ \ldots]
$$

Then compute:

$$
C_v = (\bar E_v \bar E_v^T)^\dagger
$$

$$
W_v = I_v - \bar E_v^T C_v \bar E_v
$$

Store `W_v` together with the stencil order used to create it.

Validation:

- Check `W_v` shape is `(n_v, n_v)`.
- Check `W_v` is symmetric within tolerance.
- Check `W_v` is idempotent within tolerance: `W_v @ W_v ≈ W_v`.
- Check original stencil residual is near zero: `Ebar_v @ W_v ≈ 0`.
- Apply a random affine transformation to the original stencil and verify the residual is still near zero.
- Plot residual norms over vertices using matplotlib.

### Step 7: Sparse stencil matrix `B_v` construction

Construct the sparse edge-extraction matrix `B_v` for each stencil, or implement an equivalent direct sparse assembly path.

Mathematically:

$$
B_v \in \mathbb{R}^{n_v \times n_g}
$$

Each row corresponds to one stencil vertex `i ∈ N_v`, with `+1` at column `i` and `-1` at column `v`.

Validation:

- For a random test vertex-position matrix `X`, verify `B_v X` equals the manually computed edge matrix `[x_i - x_v]`.
- Check sparse matrix shapes and nonzero counts.
- For a selected vertex, print or inspect the nonzero pattern of `B_v`.

### Step 8: Initial warp

Use the initial source binding to compute initial candidate vertices `Z0` on the target body.

Process:

1. Reconstruct target anchor points using target body vertices, source face IDs, and source barycentric coordinates.
2. Compute source and target normals at corresponding body points.
3. Rotate each original displacement vector from source normal orientation to target normal orientation.
4. Add the reoriented displacement to the target anchor point to get `z_v`.

Use normal-aligned rotation for this step. Do not use surface parallel transport here.

Validation:

- If `S` and `T` are identical, then `Z0` should match the original garment vertices within tolerance.
- Displacement lengths should be preserved by the rotation.
- Polyscope visualization: show source garment, target body, initial warped garment, target anchor points, and displacement vectors.

### Step 9: Relaxation matrix assembly

Assemble the sparse system matrix:

$$
H = \sum_v B_v^T W_v^T W_v B_v + M
$$

Implementation may explicitly construct each `B_v`, or directly accumulate local contributions into a global sparse matrix. Direct local assembly is acceptable as long as it matches the formula.

Validation:

- Check `H` has shape `(n_g, n_g)`.
- Check `H` is symmetric within tolerance.
- Check `y^T H y > 0` for several random nonzero vectors `y`, since uniform positive tightness should make `H` positive definite.
- Verify sparse factorization succeeds.
- Plot sparsity pattern for a small mesh using matplotlib.

### Step 10: Relaxation solve

Given candidate vertices `Z`, solve:

$$
H X = MZ
$$

The same matrix `H` is used for all three coordinates, so solve against a dense right-hand side of shape `(n_g, 3)`.

Validation:

- Output has shape `(n_g, 3)` and finite values.
- With the stencil term disabled, the solution should equal `Z`.
- With `S == T` and `Z` equal to the original garment, the solution should stay close to the original garment.
- Increasing the tightness weight should pull the solution closer to `Z`.
- Polyscope visualization: compare `Z` and relaxed `X`, with displacement vectors from `Z` to `X`.

### Step 11: Rebinding with normal-aligned fallback

Implement rebinding using the simpler normal-aligned displacement reorientation first.

Process:

1. Given relaxed garment vertices `X`, find new closest points on target body using Step 1.
2. Use the rebinding closest points and normals to reorient the displacement vectors.
3. Produce updated candidate vertices `Z_next`.

For the first version of rebinding, normal-aligned rotation is acceptable. Keep the API structured so `potpourri3d` transport can replace this later.

Validation:

- `Z_next` has shape `(n_g, 3)` and finite values.
- New closest-point bindings pass Step 1 validation.
- If closest points and normals do not change, candidates should remain stable.
- Polyscope visualization: show relaxed garment, target body, rebound closest points, and updated candidates.

### Step 12: Outer optimization loop

Implement the alternating optimization loop:

```text
Z = Z0
for iter in range(max_iters):
    X_new = solve_relaxation(H, M, Z)
    movement = max norm of X_new - X_previous, or use first iteration movement from Z0
    if movement < tolerance:
        break
    Z = rebind(X_new)
```

Use:

- maximum iteration count: `25`,
- tolerance: `0.0001 * bbox_size`, corresponding to `0.01%` of model bounding-box size.

Store debug history:

- iteration index,
- max movement,
- mean movement,
- closest-point distance statistics after rebinding,
- optional energy values.

Validation:

- Loop stops at or before `25` iterations.
- Movement history is finite.
- Identical source/target case should converge immediately or nearly immediately.
- Plot max and mean movement over iterations using matplotlib.
- Polyscope visualization: inspect the garment after each iteration or save selected iteration snapshots.

### Step 13: Replace rebinding fallback with `potpourri3d` transport

Implement direct parallel transport on the target body using `potpourri3d.transport_tangent_vector`.

For each displacement:

1. Decompose into tangent and normal components at the old target-body anchor.
2. Transport the tangent component from the old anchor to the new closest point.
3. Carry the scalar normal component along the new target-body normal.
4. Recombine to get the transported displacement.

Keep normal-aligned transport as a fallback option.

Validation:

- Pure normal displacement remains normal after transport.
- Pure tangent displacement remains tangent after transport.
- If old and new points are the same or very close, transported displacement should be nearly unchanged.
- Compare normal-aligned and potpourri3d-transport results visually in Polyscope.
- Plot displacement-length changes using matplotlib.

### Step 14: Top-level API

Implement a top-level function:

```python
refit_garment(
    garment_vertices,
    garment_faces,
    source_body_vertices,
    source_body_faces,
    target_body_vertices,
    target_body_faces,
    options,
) -> RefitResult
```

`RefitResult` should include:

- final refitted vertices,
- initial warped vertices,
- debug history,
- source and target bindings,
- stencils,
- optional intermediate iteration vertices.

Validation:

- API accepts the expected matrix shapes and dtypes.
- API returns final vertices with the same shape as the input garment vertices.
- API exposes enough debug data to inspect every major stage.

### Step 15: Integration tests

Keep integration tests until the end, after the pieces above are implemented.

Recommended tests:

- Tiny synthetic mesh test: simple triangle or square body patch and simple garment points.
- Identity test: `S == T`, output should remain close to original garment.
- Simple scale/shape-change test: target body is a controlled deformation of source body.
- Multi-layer stencil test: multiple garment vertices bind to the same body face and should be mutually included in stencils.
- Regression test: run the full pipeline and assert finite output, bounded movement, and stable iteration count.

Recommended visual checks:

- Polyscope scene for source body + original garment.
- Polyscope scene for target body + initial warp.
- Polyscope scene for target body + final result.
- Matplotlib plots for closest-point distances, stencil sizes, affine residual norms, relaxation matrix sparsity, and convergence curves.

