# Garment Refitting Implementation Decisions

## Pre-processing

### Closest point on body

- Use `igl.point_mesh_squared_distance` for closest-point-on-triangle-mesh queries in the first version.
- Other options considered:
  - Open3D `RaycastingScene`
  - `trimesh.proximity`

### Tightness weights

- Use uniform tightness weights for the first version.

### Garment vertex stencils

- For the first version, use two sets of vertices for each stencil \$N\_v\$:
  - One-ring garment neighbors around vertex \$v\$, without assuming \$v\$ is manifold.
  - Multi-layer / fold-over cluster augmentation: group garment vertices whose closest body point lies on the same closest body face, and add vertices in the same closest-face cluster to each other’s stencil.
- The affine-invariant stencil term does not preserve rigid distances, but it can preserve broader affine structure. For multi-layer or fold-over stencils, this is useful because it can preserve the ordering/relative arrangement of nearby layers.
- UV panel-boundary augmentation: the likely purpose is to encourage panel boundaries to transform affinely, helping retain the authored garment layout. Skip this in the first version.
- Edge/vertex closest-point caveat: if a garment vertex’s closest point on the body lies exactly on a body edge or body vertex, it may conceptually correspond to multiple adjacent body faces, and therefore perhaps multiple closest-face clusters. Ignore this in the first version and assume each garment vertex corresponds to exactly one body face.

### Affine-invariant matrix / coordinates

- For each garment vertex `v`, build the original stencil edge matrix:

  \$\bar E\_v = [\ldots,\ \bar x\_i - \bar x\_v,\ \ldots]\$

  where `i` ranges over the stencil vertices in `N_v`.

- Define:

  \$C\_v = (\bar E\_v \bar E\_v^T)^\dagger\$

  \$W\_v = I\_v - \bar E\_v^T C\_v \bar E\_v\$

  where `†` denotes the pseudoinverse and `I_v` is the identity matrix of size `n_v \times n_v`.

- Interpretation: `W_v` projects stencil coordinates onto the component that cannot be explained by a local affine transformation of the original stencil. The relaxation energy using `E_v W_v` therefore penalizes the non-affine component of the current/deformed stencil.

### Body frame field / displacement reorientation

- Current top option: use `potpourri3d` `transport_tangent_vector` for tangent-vector transport: [https://github.com/nmwsharp/potpourri3d#polygon-mesh-distance--transport](https://github.com/nmwsharp/potpourri3d#polygon-mesh-distance--transport)
- Because `transport_tangent_vector` transports tangent vectors, decompose each displacement into tangent and normal components before transport. Transport the tangent component; carry the normal component using the corresponding body normal.
- This performs parallel transport without explicitly constructing a global directional/frame field. This differs from the paper’s frame-field approach, but may be more accurate for actual vector transport because it directly solves the transport problem instead of relying on a precomputed field convention.
- Potential downside: it may be slower than using already-computed frames for repeated rebinding steps, though the GitHub notes suggest it becomes faster after the initial computation.
- Simpler first-version possibility: use naive normal-aligned transport, i.e. rotate the displacement vector by the smallest rotation aligning the old body normal/frame to the new body normal/frame.
- Backup options:
  - Use a combed principal-curvature field from libigl, e.g. compute principal directions and comb them with `comb_line_field` or `comb_cross_field`.
  - Make a Python binding for `igl::nrosy`.
  - Make a Python binding for Directional: [https://github.com/avaxman/Directional](https://github.com/avaxman/Directional)

## Initial warp

- Use barycentric coordinates from the source-body closest point to reconstruct the corresponding body point on the target body. This relies on source and target bodies sharing mesh connectivity.
- For initial warp, do not use parallel transport, since the source and target are different geometries rather than two points on the same surface.
- Reorient the displacement vector by rotating it according to the source-to-target normal orientation, then add it to the reconstructed target-body point to get the candidate garment vertex.

## Optimization

- Termination criteria: stop when the largest garment vertex movement is less than 0.01% of the model bounding-box size, or when 25 iterations have been reached.

### Relaxation

- Define a sparse stencil edge-extraction matrix:

  $$B_v \in \mathbb{R}^{n_v \times n_g}$$

  where $n_v$ is the number of stencil vertices for garment vertex $v$, and $n_g$ is the total number of garment vertices.
- Each row of $B_v$ corresponds to one stencil vertex $i \in N_v$: it has $+1$ at stencil vertex $i$, $-1$ at center vertex $v$, and zeros elsewhere. Thus $B_v X$ gives the current stencil edge vectors $x_i - x_v$.
- The relaxation system matrix has the form:

  $$H = \sum_v B_v^T W_v^T W_v B_v + M$$

  where $M = \operatorname{diag}(m_v)$ is the tightness-weight matrix.
- The relaxation solve is:

  $$H X = R$$

  with right-hand side:

  $$R = M Z$$

  where $Z$ stores the current candidate garment vertices $z_v$.
- $H$ is positive semidefinite because it is a sum of squared linear residual terms. With uniform positive tightness weights, $M$ is positive definite, so $H$ becomes positive definite.
- There are ideas in the chat for weighting vertices within a stencil, but leave those out of the implementation note for now and revisit later.

### Rebinding

- Step purpose: find new closest points on the target body \$T\$, parallel-transport the displacement vectors, then obtain updated candidate vertices \$z\_v\$.
- Refer to the earlier closest-point section for how to find closest points on the body during rebinding.
- Use direct parallel transport for displacement-vector reorientation rather than constructing an explicit frame field.

## Post-processing

- The paper uses cloth simulation as a post-processing step to resolve garment/body intersections and add draping details.
- Skip post-processing for this implementation.
