from __future__ import annotations

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from example_data import DEFAULT_SOURCE_SET_ID, DEFAULT_TARGET_SET_ID, list_mesh_set_ids, load_mesh_set
from refitting import GarmentRefittingManager


REBINDING_METHODS = ["directional_field", "normal_aligned"]


def main() -> None:
    set_ids = list_mesh_set_ids()
    source_set_index = set_ids.index(DEFAULT_SOURCE_SET_ID)
    target_set_index = set_ids.index(DEFAULT_TARGET_SET_ID)
    rebinding_method_index = 0

    manager = build_manager(
        set_ids[source_set_index],
        set_ids[target_set_index],
        REBINDING_METHODS[rebinding_method_index],
    )

    target_shift = np.zeros(3, dtype=np.float32)
    source_garment = None
    warped_garment = None
    relaxation_displacements = None
    rebound_points = None
    rebound_displacements = None

    tightness_weight = manager.tightness_weight
    garment_translucent = True
    auto_run_until_converged = False
    show_max_iteration_warning = False

    # Use explicit centimeter units from the data, so point/curve radii are not scene-relative.
    ps.set_autoscale_structures(False)
    ps.init()
    ps.set_automatically_compute_scene_extents(False)

    def register_scene() -> None:
        nonlocal target_shift
        nonlocal source_garment, warped_garment, relaxation_displacements, rebound_points, rebound_displacements

        ps.remove_all_structures()

        source_body_vertices_np = manager.source_body_vertices.numpy()
        source_body_faces_np = manager.source_body_faces.numpy()
        source_garment_vertices_np = manager.garment_vertices.numpy()
        source_garment_faces_np = manager.garment_faces.numpy()
        target_body_vertices_np = manager.target_body_vertices.numpy()
        target_body_faces_np = manager.target_body_faces.numpy()
        source_closest_points_np = manager.initial_warp.source_binding.closest_points.numpy()
        target_anchor_points_np = manager.initial_warp.target_anchor_points.numpy()
        candidate_vertices_np = manager.current_candidate_vertices.numpy()

        source_points = np.vstack([source_body_vertices_np, source_garment_vertices_np])
        target_points = np.vstack([target_body_vertices_np, candidate_vertices_np])
        source_width = source_points[:, 0].max() - source_points[:, 0].min()
        x_offset = source_width + 10.0
        target_shift = np.array([x_offset, 0.0, 0.0], dtype=np.float32)

        shifted_target_body_vertices_np = target_body_vertices_np + target_shift
        shifted_target_anchor_points_np = target_anchor_points_np + target_shift
        shifted_candidate_vertices_np = candidate_vertices_np + target_shift

        scene_points = np.vstack(
            [
                source_body_vertices_np,
                source_garment_vertices_np,
                shifted_target_body_vertices_np,
                shifted_candidate_vertices_np,
            ]
        )
        bbox_low = scene_points.min(axis=0)
        bbox_high = scene_points.max(axis=0)
        ps.set_bounding_box(bbox_low, bbox_high)
        ps.set_length_scale(np.linalg.norm(bbox_high - bbox_low))

        ps.register_surface_mesh("source body", source_body_vertices_np, source_body_faces_np)
        source_garment = ps.register_surface_mesh(
            "source garment",
            source_garment_vertices_np,
            source_garment_faces_np,
            transparency=0.45 if garment_translucent else 1.0,
        )
        ps.register_surface_mesh(
            "target body",
            shifted_target_body_vertices_np,
            target_body_faces_np,
        )
        warped_garment = ps.register_surface_mesh(
            "warped garment",
            shifted_candidate_vertices_np,
            source_garment_faces_np,
            transparency=0.45 if garment_translucent else 1.0,
        )

        source_closest_points = ps.register_point_cloud("source closest body points", source_closest_points_np)
        source_closest_points.set_radius(0.1, relative=False)

        source_segments = source_closest_points_np.repeat(2, axis=0)
        source_segments[1::2] = source_garment_vertices_np
        source_edges = np.arange(source_garment_vertices_np.shape[0] * 2, dtype=np.int32).reshape(-1, 2)
        source_displacements = ps.register_curve_network("source binding displacements", source_segments, source_edges)
        source_displacements.set_radius(0.03, relative=False)

        anchors = ps.register_point_cloud("target anchor points", shifted_target_anchor_points_np)
        anchors.set_radius(0.1, relative=False)

        segments = shifted_target_anchor_points_np.repeat(2, axis=0)
        segments[1::2] = shifted_candidate_vertices_np
        edges = np.arange(candidate_vertices_np.shape[0] * 2, dtype=np.int32).reshape(-1, 2)
        displacements = ps.register_curve_network("reoriented displacements", segments, edges)
        displacements.set_radius(0.03, relative=False)

        relaxed_segments = shifted_candidate_vertices_np.repeat(2, axis=0)
        relaxed_edges = np.arange(candidate_vertices_np.shape[0] * 2, dtype=np.int32).reshape(-1, 2)
        relaxation_displacements = ps.register_curve_network(
            "relaxation displacements",
            relaxed_segments,
            relaxed_edges,
            enabled=False,
        )
        relaxation_displacements.set_radius(0.03, relative=False)

        rebound_points = ps.register_point_cloud(
            "rebound closest points",
            shifted_target_anchor_points_np,
            enabled=False,
        )
        rebound_points.set_radius(0.1, relative=False)

        rebound_segments = shifted_candidate_vertices_np.repeat(2, axis=0)
        rebound_displacements = ps.register_curve_network(
            "rebound displacements",
            rebound_segments,
            relaxed_edges,
            enabled=False,
        )
        rebound_displacements.set_radius(0.03, relative=False)

    def reset_iteration_display() -> None:
        manager.reset()
        shifted_candidate_vertices_np = manager.current_candidate_vertices.numpy() + target_shift
        warped_garment.update_vertex_positions(shifted_candidate_vertices_np)
        relaxation_displacements.set_enabled(False)
        rebound_points.set_enabled(False)
        rebound_displacements.set_enabled(False)

    def update_relaxation_display(previous_vertices, relaxed_vertices) -> None:
        shifted_previous_vertices_np = previous_vertices.numpy() + target_shift
        shifted_relaxed_vertices_np = relaxed_vertices.numpy() + target_shift
        warped_garment.update_vertex_positions(shifted_relaxed_vertices_np)

        relaxation_segments = shifted_previous_vertices_np.repeat(2, axis=0)
        relaxation_segments[1::2] = shifted_relaxed_vertices_np
        relaxation_displacements.update_node_positions(relaxation_segments)
        relaxation_displacements.set_enabled(True)

    def update_rebinding_display(rebinding) -> None:
        shifted_rebound_points_np = rebinding.target_binding.closest_points.numpy() + target_shift
        shifted_rebound_candidates_np = rebinding.candidate_vertices.numpy() + target_shift
        warped_garment.update_vertex_positions(shifted_rebound_candidates_np)
        rebound_points.update_point_positions(shifted_rebound_points_np)
        rebound_points.set_enabled(True)

        rebound_segments = shifted_rebound_points_np.repeat(2, axis=0)
        rebound_segments[1::2] = shifted_rebound_candidates_np
        rebound_displacements.update_node_positions(rebound_segments)
        rebound_displacements.set_enabled(True)

    def callback() -> None:
        nonlocal auto_run_until_converged, garment_translucent, manager, rebinding_method_index, tightness_weight
        nonlocal show_max_iteration_warning, source_set_index, target_set_index

        changed, source_set_index = psim.Combo("source set", source_set_index, set_ids)
        if changed:
            manager = build_manager(
                set_ids[source_set_index],
                set_ids[target_set_index],
                REBINDING_METHODS[rebinding_method_index],
                tightness_weight,
            )
            register_scene()
            auto_run_until_converged = False
            show_max_iteration_warning = False

        changed, target_set_index = psim.Combo("target set", target_set_index, set_ids)
        if changed:
            target_body_vertices, target_body_faces, _, _ = load_mesh_set(set_ids[target_set_index])
            manager.change_target_body(target_body_vertices, target_body_faces)
            register_scene()
            auto_run_until_converged = False
            show_max_iteration_warning = False

        changed, garment_translucent = psim.Checkbox("translucent garments", garment_translucent)
        if changed:
            transparency = 0.45 if garment_translucent else 1.0
            source_garment.set_transparency(transparency)
            warped_garment.set_transparency(transparency)

        changed, rebinding_method_index = psim.Combo(
            "rebinding method",
            rebinding_method_index,
            REBINDING_METHODS,
        )
        if changed:
            manager.rebinding_method = REBINDING_METHODS[rebinding_method_index]
            auto_run_until_converged = False

        changed, tightness_weight = psim.InputFloat(
            "tightness weight",
            tightness_weight,
            step=0.1,
            step_fast=1.0,
            format="%.4f",
        )
        tightness_weight = max(tightness_weight, 1e-8)
        if changed:
            manager.change_tightness_weight(tightness_weight)
            register_scene()
            auto_run_until_converged = False
            show_max_iteration_warning = False

        if psim.Button("run one relaxation iteration"):
            previous_vertices = manager.current_candidate_vertices
            relaxed_vertices = manager.run_relaxation_step()
            update_relaxation_display(previous_vertices, relaxed_vertices)

        if psim.Button("run rebinding"):
            rebinding = manager.run_rebinding_step()
            update_rebinding_display(rebinding)

        if psim.Button("run until converged"):
            auto_run_until_converged = True
            show_max_iteration_warning = False

        if auto_run_until_converged and not manager.converged:
            previous_vertices = manager.current_candidate_vertices
            stats = manager.run_iteration()
            update_relaxation_display(previous_vertices, manager.current_relaxed_vertices)
            shifted_rebound_points_np = manager.last_rebinding.target_binding.closest_points.numpy() + target_shift
            shifted_rebound_candidates_np = manager.last_rebinding.candidate_vertices.numpy() + target_shift
            rebound_points.update_point_positions(shifted_rebound_points_np)
            rebound_points.set_enabled(True)

            rebound_segments = shifted_rebound_points_np.repeat(2, axis=0)
            rebound_segments[1::2] = shifted_rebound_candidates_np
            rebound_displacements.update_node_positions(rebound_segments)
            rebound_displacements.set_enabled(True)
            if manager.converged:
                auto_run_until_converged = False
                show_max_iteration_warning = not stats.converged

        if psim.Button("reset"):
            reset_iteration_display()
            auto_run_until_converged = False
            show_max_iteration_warning = False

        if show_max_iteration_warning:
            psim.TextColored((1.0, 0.3, 0.2, 1.0), "Warning: reached max iterations before convergence.")

    register_scene()
    ps.set_user_callback(callback)
    ps.show()


def build_manager(
    source_set_id: str,
    target_set_id: str,
    rebinding_method: str,
    tightness_weight: float = 0.1,
) -> GarmentRefittingManager:
    source_body_vertices, source_body_faces, source_garment_vertices, source_garment_faces = load_mesh_set(source_set_id)
    target_body_vertices, target_body_faces, _, _ = load_mesh_set(target_set_id)
    return GarmentRefittingManager(
        source_garment_vertices,
        source_garment_faces,
        source_body_vertices,
        source_body_faces,
        target_body_vertices,
        target_body_faces,
        tightness_weight=tightness_weight,
        rebinding_method=rebinding_method,
    )


if __name__ == "__main__":
    main()
