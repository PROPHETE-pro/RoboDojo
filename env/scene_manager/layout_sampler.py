"""Online task-layout sampling (RoboTwin load_actors analogue).

Builds an in-memory dict with the same schema as Assets/Eval_Layout JSON.
Does not write files. Domain randomization (table/ground/background/light)
is not applied; fixtures come from the scene YAML defaults.
"""

from __future__ import annotations

import os
import random
from copy import deepcopy
from typing import Any

import numpy as np
import transforms3d as t3d
from shapely.geometry import box
from shapely.ops import unary_union

from env.global_configs import OBJECTS_PATH
from utils.cluttered_generator import ClutteredGenerator, UnStableError
from utils.load_file import load_object_metadata, load_yaml
from utils.transformer import rotate_quat_about_world_axis

_VARIANT_KEYS = (
    "place_tag",
    "xlim",
    "ylim",
    "zlim",
    "rotate_rand",
    "rotate_deg",
    "qpos",
    "margin",
    "check_mode",
    "need_check_stable",
)

OBJECT_SPAWN_ORDER = ("Geometry", "Articulation", "Rigid", "Dynamic", "Garment", "Fluid")
PHYSICS_TYPE = {
    "Rigid": "rigid",
    "Dynamic": "dynamic",
    "Geometry": "geometry",
    "Articulation": "articulation",
    "Garment": "garment",
    "Fluid": "fluid",
}


def generate_layout(
    task_config: dict[str, Any],
    scene_config: dict[str, Any],
    env_spacing: float = 2.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Sample one Eval_Layout-isomorphic scene from task YAML constraints."""
    if seed is not None:
        _seed_rng(int(seed))
    task_config = _to_plain(task_config)
    scene_config = _to_plain(scene_config)
    task_config = _merge_scene_objects(task_config, scene_config)

    table_info = _table_info(scene_config)
    ground_info = _ground_info(scene_config, env_spacing)
    generators = {
        "Table": _make_plane_generator(table_info),
        "Ground": _make_plane_generator(ground_info),
    }
    _apply_prohibited_areas(generators, task_config.get("ProhibitedArea"))

    layout: dict[str, Any] = {}
    placed: dict[str, dict[str, Any]] = {}
    used_parents: set[str] = set()
    # Select asset indices in YAML order so same_index_as_label can see labels
    # that are spawned later (Geometry bolts reference Rigid nuts).
    assignments = _preselect_placements(task_config)

    for object_type in OBJECT_SPAWN_ORDER:
        groups = task_config.get(object_type) or []
        if not groups:
            continue
        type_bucket: dict[str, list[dict[str, Any]]] = {}
        for group_index, group in enumerate(groups):
            attempts = assignments.get((object_type, group_index))
            if not attempts:
                continue
            for inst in _sample_group(
                attempts=attempts,
                object_type=object_type,
                generators=generators,
                table_info=table_info,
                ground_info=ground_info,
                placed=placed,
                used_parents=used_parents,
                cluttered=False,
            ):
                type_bucket.setdefault(inst["category"], []).append(inst)
                if inst.get("label"):
                    placed[inst["label"]] = inst
        if type_bucket:
            layout[object_type] = type_bucket

    _sample_clutter(
        layout=layout,
        clutter_groups=task_config.get("Clutter") or [],
        generators=generators,
        table_info=table_info,
        ground_info=ground_info,
        placed=placed,
        used_parents=used_parents,
    )

    layout["Room"] = _fixture_room(scene_config)
    layout["Table"] = _fixture_table(scene_config)
    layout["Ground"] = _fixture_ground(scene_config)
    layout["Background"] = _fixture_background(scene_config)
    return layout


def _seed_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _to_plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "keys") and type(value).__name__ == "DictConfig":
        try:
            from omegaconf import OmegaConf

            return OmegaConf.to_container(value, resolve=True)
        except Exception:
            return {str(k): _to_plain(value[k]) for k in value}
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _as_int(value: Any, default: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else default
    return int(value)


def _intervals(lim: Any) -> list[tuple[float, float]]:
    if lim is None:
        return []
    if isinstance(lim, (int, float)):
        v = float(lim)
        return [(v, v)]
    seq = list(lim)
    if not seq:
        return []
    if isinstance(seq[0], (list, tuple)):
        out = []
        for item in seq:
            a, b = float(item[0]), float(item[1] if len(item) > 1 else item[0])
            out.append((min(a, b), max(a, b)))
        return out
    a, b = float(seq[0]), float(seq[1] if len(seq) > 1 else seq[0])
    return [(min(a, b), max(a, b))]


def _region_from_lims(xlim: Any, ylim: Any):
    x_iv = _intervals(xlim) or [(0.0, 0.0)]
    y_iv = _intervals(ylim) or [(0.0, 0.0)]
    parts = []
    for x0, x1 in x_iv:
        for y0, y1 in y_iv:
            parts.append(box(x0, y0, x1, y1))
    if len(parts) == 1:
        return parts[0]
    return unary_union(parts)


def _group_labels(group: dict[str, Any]) -> set[str]:
    return set((group.get("select_mode") or {}).get("label") or [])


def _merge_scene_objects(task_config: dict[str, Any], scene_config: dict[str, Any]) -> dict[str, Any]:
    """Append scene object groups (e.g. camera_stand) unless already merged."""
    merged = deepcopy(task_config)
    for key in OBJECT_SPAWN_ORDER:
        extra = scene_config.get(key) or []
        if not extra:
            continue
        current = list(merged.get(key) or [])
        existing = set()
        for group in current:
            existing |= _group_labels(group)
        for group in extra:
            labels = _group_labels(group)
            if labels and labels <= existing:
                continue
            current.append(deepcopy(group))
            existing |= labels
        merged[key] = current
    return merged


def _table_info(scene_config: dict[str, Any]) -> dict[str, Any]:
    table = scene_config.get("Table") or {}
    default_pos = list(table.get("default_pos") or [0.0, -0.05, 0.74])
    scale = list(table.get("scale") or [1.4, 1.1, 0.05])
    return {
        "pos": default_pos,
        "size": [-scale[0] / 2, -scale[1] / 2, scale[0] / 2, scale[1] / 2],
        "height": float(default_pos[2]) + float(scale[2]) / 2.0,
        "scale": scale,
    }


def _ground_info(scene_config: dict[str, Any], env_spacing: float) -> dict[str, Any]:
    ground = scene_config.get("Ground") or {}
    default_pos = list(ground.get("default_pos") or [0.0, 0.0, 0.0])
    thickness = float(ground.get("thickness", 0.1))
    half = float(env_spacing) / 2.0
    return {
        "pos": default_pos,
        "size": [-half, -half, half, half],
        "height": thickness / 2.0,
    }


def _make_plane_generator(info: dict[str, Any]) -> ClutteredGenerator:
    pos = info.get("pos") or [0.0, 0.0, 0.0]
    size = info.get("size") or [0.0, 0.0, 0.0, 0.0]
    gen = ClutteredGenerator()
    gen.reset(
        box(
            size[0] + 0.05 + pos[0],
            size[1] + 0.05 + pos[1],
            size[2] - 0.05 + pos[0],
            size[3] - 0.05 + pos[1],
        ),
        frame=np.array([0.0, 0.0, float(info.get("height", 0.0)), 1.0, 0.0, 0.0, 0.0]),
    )
    return gen


def _apply_prohibited_areas(generators: dict[str, ClutteredGenerator], items: Any) -> None:
    """Task ProhibitedArea is a table-workspace constraint, not a ground fixture mask."""
    if not items:
        return
    table_gen = generators.get("Table")
    if table_gen is None:
        return
    for i, item in enumerate(items):
        coords = item
        if isinstance(item, dict):
            coords = item.get("bbox") or item.get("box") or item.get("xyxy")
        if not isinstance(coords, (list, tuple)) or len(coords) < 4:
            continue
        minx, miny, maxx, maxy = (float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))
        table_gen.add_box_prohibited_area(minx, miny, maxx, maxy, name=f"prohibited_{i}")


def _available_indices(object_type: str, category: str, cluttered: bool = False) -> list[int]:
    modeldir = _object_dir(object_type, category, cluttered)
    if not os.path.isdir(modeldir):
        return []
    indices = []
    for name in os.listdir(modeldir):
        try:
            indices.append(int(name))
        except ValueError:
            continue
    return sorted(indices)


def _category_pool(group: dict[str, Any], object_type: str, cluttered: bool = False) -> list[tuple[str, int]]:
    pool = []
    for cat in group.get("category") or []:
        name = cat.get("name")
        if not name:
            continue
        indices = cat.get("index")
        if not indices:
            indices = _available_indices(object_type, str(name), cluttered=cluttered)
        for idx in indices:
            pool.append((str(name), int(idx)))
    return pool


def _select_instances(
    pool: list[tuple[str, int]],
    nums: int,
    mode: str,
) -> list[tuple[str, int]]:
    if not pool:
        raise UnStableError("empty category pool")
    mode = (mode or "allow_duplicate").strip()
    if mode == "same":
        picked = pool[int(np.random.randint(len(pool)))]
        return [picked] * nums
    if mode == "unique":
        if nums > len(pool):
            raise UnStableError(f"unique select needs {nums} items, pool has {len(pool)}")
        chosen = np.random.choice(len(pool), size=nums, replace=False)
        return [pool[int(i)] for i in chosen]
    if mode == "category_unique":
        by_cat: dict[str, list[int]] = {}
        for name, idx in pool:
            by_cat.setdefault(name, []).append(idx)
        cats = list(by_cat.keys())
        if nums > len(cats):
            raise UnStableError(f"category_unique needs {nums} cats, pool has {len(cats)}")
        chosen_cats = np.random.choice(len(cats), size=nums, replace=False)
        out = []
        for ci in chosen_cats:
            cat = cats[int(ci)]
            idx = by_cat[cat][int(np.random.randint(len(by_cat[cat])))]
            out.append((cat, int(idx)))
        return out
    # allow_duplicate
    chosen = np.random.randint(0, len(pool), size=nums)
    return [pool[int(i)] for i in chosen]


def _object_dir(object_type: str, category: str, cluttered: bool) -> str:
    if cluttered:
        return os.path.join(OBJECTS_PATH, "Clutter", category)
    return os.path.join(OBJECTS_PATH, object_type, category)


def _normalize_place_tag(place_tag: Any) -> list[str] | None:
    if place_tag is None:
        return None
    if isinstance(place_tag, str):
        tags = [place_tag]
        if "/" in place_tag:
            tags.append(place_tag.split("/")[0])
        return tags
    return [str(t) for t in place_tag]


def _split_relative_plane(relative_plane: Any) -> tuple[str, str | None]:
    if relative_plane is None:
        return "Table", None
    if isinstance(relative_plane, (list, tuple)):
        raise TypeError("list relative_plane must be resolved before split")
    text = str(relative_plane)
    if text in ("Table", "Ground"):
        return text, None
    parts = text.split("/")
    if len(parts) == 1:
        return parts[0], None
    return parts[0], "/".join(parts[1:])


def _pick_relative_plane(common: dict[str, Any], used_parents: set[str]) -> str:
    relative = common.get("relative_plane", "Table")
    if isinstance(relative, (list, tuple)):
        candidates = [str(x) for x in relative]
        free = [c for c in candidates if c not in used_parents]
        pool = free or candidates
        chosen = pool[int(np.random.randint(len(pool)))]
        used_parents.add(chosen)
        return chosen
    return str(relative or "Table")


def _variant_overrides(category: dict[str, Any]) -> dict[str, Any]:
    return {key: category[key] for key in _VARIANT_KEYS if key in category}


def _uses_category_variants(group: dict[str, Any], mode: str) -> bool:
    if mode != "same":
        return False
    return any(
        _variant_overrides(cat)
        for cat in (group.get("category") or [])
        if isinstance(cat, dict)
    )


def _indices_for_category(
    category: dict[str, Any],
    object_type: str,
    cluttered: bool = False,
) -> list[tuple[str, int]]:
    name = category.get("name")
    if not name:
        return []
    indices = category.get("index")
    if not indices:
        indices = _available_indices(object_type, str(name), cluttered=cluttered)
    return [(str(name), int(idx)) for idx in indices]


def _compose_pose(parent_pose: np.ndarray, local_pose: np.ndarray) -> np.ndarray:
    """Map a pose in the parent frame into the parent pose's frame.

    ``R_world = R_parent @ R_local``, matching ``LayoutManager.get_support_points``
    (``ref_matrix @ pose_to_matrix``). Position uses the parent rotation only.
    """
    parent_pose = np.asarray(parent_pose, dtype=float).reshape(7)
    local_pose = np.asarray(local_pose, dtype=float).reshape(7)
    world = np.zeros(7, dtype=float)
    world[:3] = parent_pose[:3] + t3d.quaternions.rotate_vector(local_pose[:3], parent_pose[3:])
    world[3:] = t3d.quaternions.qmult(parent_pose[3:], local_pose[3:])
    return world


def _origin_from_contact(contact_world: np.ndarray, contact_local: np.ndarray) -> np.ndarray:
    """Invert ``_compose_pose``: contact = origin ∘ contact_local."""
    contact_world = np.asarray(contact_world, dtype=float).reshape(7)
    contact_local = np.asarray(contact_local, dtype=float).reshape(7)
    origin = np.zeros(7, dtype=float)
    origin[3:] = t3d.quaternions.qmult(
        contact_world[3:], t3d.quaternions.qinverse(contact_local[3:])
    )
    origin[:3] = contact_world[:3] - t3d.quaternions.rotate_vector(contact_local[:3], origin[3:])
    return origin


def _bbox_vertices(metadata: dict[str, Any] | None) -> np.ndarray | None:
    vertices = ((metadata or {}).get("geometry") or {}).get("oriented_bbox", {}).get("vertices")
    if not vertices:
        return None
    return np.asarray(vertices, dtype=float).reshape(-1, 3)


def _world_min_z(pose: np.ndarray, vertices: np.ndarray) -> float:
    rot = t3d.quaternions.quat2mat(np.asarray(pose[3:], dtype=float))
    return float((vertices @ rot.T + np.asarray(pose[:3], dtype=float))[:, 2].min())


def _lift_above_parent(
    origin: np.ndarray,
    child_metadata: dict[str, Any],
    parent_pose: np.ndarray,
    parent_metadata: dict[str, Any] | None,
    clearance: float = 0.002,
) -> np.ndarray:
    """Raise a support-snapped child so its mesh does not start inside the table.

    Support centers are often the grasp/slot anchor, which can sit inside the
    child mesh. A vertical coin centered on that anchor clips the table and the
    stability check rejects every seed.
    """
    child_vertices = _bbox_vertices(child_metadata)
    parent_vertices = _bbox_vertices(parent_metadata)
    if child_vertices is None or parent_vertices is None:
        return origin
    gap = _world_min_z(parent_pose, parent_vertices) + clearance - _world_min_z(origin, child_vertices)
    if gap <= 0.0:
        return origin
    lifted = np.array(origin, dtype=float, copy=True)
    lifted[2] += gap
    return lifted


def _record_pose(record: dict[str, Any]) -> np.ndarray:
    pos = record.get("default_pos") or [0.0, 0.0, 0.0]
    ori = record.get("default_ori") or [1.0, 0.0, 0.0, 0.0]
    return np.asarray([*pos[:3], *ori[:4]], dtype=float)


def _object_type_of(record: dict[str, Any]) -> str:
    physics_type = str((record.get("physics") or {}).get("type") or "rigid")
    for name, value in PHYSICS_TYPE.items():
        if value == physics_type:
            return name
    return "Rigid"


def _support_local_pose(parent: dict[str, Any] | None, support_key: str | None) -> np.ndarray | None:
    """Return one ``passive.support`` center of ``parent``, or None if it is not a support."""
    if parent is None or not support_key:
        return None
    metadata = load_object_metadata(
        _object_dir(_object_type_of(parent), str(parent.get("category")), cluttered=False),
        int(parent.get("category_idx", 0)),
    )
    if not metadata:
        return None
    item = ((metadata.get("passive") or {}).get("support") or {}).get(support_key)
    if not isinstance(item, dict):
        return None
    centers = item.get("center") or []
    if not centers:
        return None
    center = centers[int(np.random.randint(len(centers)))]
    return np.asarray(center, dtype=float).reshape(7)


def _local_xy(xlim: Any, ylim: Any) -> tuple[float, float]:
    xs = _intervals(xlim) or [(0.0, 0.0)]
    ys = _intervals(ylim) or [(0.0, 0.0)]
    point_like = (
        len(xs) == 1
        and len(ys) == 1
        and abs(xs[0][0] - xs[0][1]) <= 1e-12
        and abs(ys[0][0] - ys[0][1]) <= 1e-12
    )
    if point_like:
        return float(xs[0][0]), float(ys[0][0])
    gen = ClutteredGenerator()
    region = gen._normalize_region(_region_from_lims(xlim, ylim))
    sampled = gen.sample_point_in_region(region)
    if sampled is None:
        return (float(xs[0][0]) + float(xs[0][1])) / 2.0, (float(ys[0][0]) + float(ys[0][1])) / 2.0
    return float(sampled[0]), float(sampled[1])


def _contact_frames(metadata: dict[str, Any], place_tag: Any) -> list[np.ndarray]:
    places = ((metadata.get("active") or {}).get("place") or {})
    allowed = None if place_tag is None else set(_normalize_place_tag(place_tag) or [])
    frames: list[np.ndarray] = []
    for key, data in places.items():
        if allowed is not None and key not in allowed:
            continue
        center = ((data or {}).get("projection_circle") or {}).get("center")
        if center is None:
            continue
        frames.append(np.asarray(center, dtype=float).reshape(7))
    if not frames and allowed is not None:
        return _contact_frames(metadata, None)
    if not frames:
        frames.append(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float))
    return frames


def _pose_on_support(
    parent: dict[str, Any],
    support_local: np.ndarray,
    metadata: dict[str, Any],
    common: dict[str, Any],
) -> np.ndarray:
    """Place the child contact on the parent support point.

    World pose is parent ∘ support ∘ xlim/ylim offset. The child's own place tag
    converts that contact pose into the object origin. The support key is not a
    child place tag.
    """
    dx, dy = _local_xy(common.get("xlim"), common.get("ylim"))
    offset = np.array([dx, dy, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    parent_pose = _record_pose(parent)
    contact_world = _compose_pose(parent_pose, _compose_pose(support_local, offset))
    frames = _contact_frames(metadata, common.get("place_tag"))
    contact_trans = frames[int(np.random.randint(len(frames)))]
    gen = ClutteredGenerator()
    origin = _origin_from_contact(contact_world, contact_trans)
    if common.get("rotate_rand"):
        angle = gen._sample_rotate_angle(common.get("rotate_deg"))
        origin[3:] = rotate_quat_about_world_axis(
            origin[3:], np.array([0.0, 0.0, 1.0]), angle_deg=angle
        )
    parent_metadata = load_object_metadata(
        _object_dir(_object_type_of(parent), str(parent.get("category")), cluttered=False),
        int(parent.get("category_idx", 0)),
    )
    return _lift_above_parent(origin, metadata, parent_pose, parent_metadata)


def _footprint_blocked(table_gen: ClutteredGenerator, polygon, parent_label: str) -> bool:
    candidate_ids = list(table_gen.rtree_idx.intersection(polygon.bounds))
    for pid in candidate_ids:
        other_name, other = table_gen.placed_polygons[pid]
        if other_name == parent_label or str(other_name).startswith(f"{parent_label}_"):
            continue
        if polygon.intersects(other) and not polygon.touches(other):
            return True
    return False


def _reserve_support_footprint(
    table_gen: ClutteredGenerator,
    pose: np.ndarray,
    metadata: dict[str, Any],
    common: dict[str, Any],
    name: str,
    parent_label: str,
) -> bool:
    """Record the child footprint on the table. ``point`` mode keeps the support pose."""
    vertices = ((metadata.get("geometry") or {}).get("oriented_bbox") or {}).get("vertices")
    if vertices is None:
        return True
    polygon, _z_max = table_gen._calc_polygon(
        np.asarray(pose, dtype=float),
        np.asarray(vertices, dtype=float),
        float(common.get("margin", 0.01)),
    )
    check_mode = str(common.get("check_mode", "bbox"))
    blocked = check_mode != "enforce" and _footprint_blocked(table_gen, polygon, parent_label)
    if blocked and check_mode != "point":
        return False
    table_gen.add_polygon(polygon, name=name or "model", check_mode="enforce")
    return True


def _ordered_task_groups(task_config: dict[str, Any]) -> list[tuple[str, int, dict[str, Any]]]:
    """Groups in YAML key order. The index matches ``task_config[object_type]``."""
    ordered = []
    for key, value in task_config.items():
        if key not in OBJECT_SPAWN_ORDER or not isinstance(value, list):
            continue
        for group_index, group in enumerate(value):
            if isinstance(group, dict):
                ordered.append((key, group_index, group))
    return ordered


def _pack_choices(
    picks: list[tuple[str, int]],
    labels: list[Any],
    common: dict[str, Any],
) -> list[tuple[str, int, str, dict[str, Any]]]:
    choices = []
    for i, (category, category_idx) in enumerate(picks):
        label = labels[i] if i < len(labels) else f"{category}{i}"
        choices.append((str(category), int(category_idx), str(label), dict(common)))
    return choices


def _choose_group_placements(
    group: dict[str, Any],
    object_type: str,
    selected: dict[str, tuple[str, int]],
) -> list[list[tuple[str, int, str, dict[str, Any]]]]:
    """Return placement attempts. Variant groups list every pose, shuffled."""
    common = dict(group.get("common") or {})
    select = group.get("select_mode") or {}
    nums = _as_int(select.get("nums", 1), 1)
    mode = str(select.get("mode", "allow_duplicate")).strip()
    labels = list(select.get("label") or [])
    if mode == "same_index_as_label":
        same_label = select.get("same_label")
        ref = selected.get(str(same_label))
        if ref is None:
            raise UnStableError(f"same_index_as_label missing label={same_label}")
        pool = _category_pool(group, object_type, cluttered=False)
        matches = [(name, idx) for name, idx in pool if int(idx) == int(ref[1])]
        if not matches:
            raise UnStableError(
                f"same_index_as_label label={same_label} index={ref[1]} not in {object_type} pool"
            )
        return [_pack_choices([matches[0]] * nums, labels, common)]
    if _uses_category_variants(group, mode):
        variants = [
            cat for cat in (group.get("category") or []) if isinstance(cat, dict) and cat.get("name")
        ]
        order = np.random.permutation(len(variants))
        attempts = []
        for variant_index in order:
            variant = variants[int(variant_index)]
            variant_common = dict(common)
            variant_common.update(_variant_overrides(variant))
            picks = _select_instances(_indices_for_category(variant, object_type), nums, "same")
            attempts.append(_pack_choices(picks, labels, variant_common))
        return attempts
    picks = _select_instances(_category_pool(group, object_type, cluttered=False), nums, mode)
    return [_pack_choices(picks, labels, common)]


def _preselect_placements(
    task_config: dict[str, Any],
) -> dict[tuple[str, int], list[list[tuple[str, int, str, dict[str, Any]]]]]:
    selected: dict[str, tuple[str, int]] = {}
    assignments: dict[tuple[str, int], list[list[tuple[str, int, str, dict[str, Any]]]]] = {}
    for object_type, group_index, group in _ordered_task_groups(task_config):
        attempts = _choose_group_placements(group, object_type, selected)
        assignments[(object_type, group_index)] = attempts
        for category, category_idx, label, _common in attempts[0]:
            if label:
                selected[label] = (category, category_idx)
    return assignments


def _generator_for_plane(
    plane: str,
    generators: dict[str, ClutteredGenerator],
    placed: dict[str, dict[str, Any]],
    table_info: dict[str, Any],
    ground_info: dict[str, Any],
    xlim: Any,
    ylim: Any,
) -> tuple[ClutteredGenerator, Any, list[str] | None]:
    label, extra_tag = _split_relative_plane(plane)
    if label in ("Table", "Ground"):
        return generators[label], _region_from_lims(xlim, ylim), _normalize_place_tag(extra_tag)
    parent = placed.get(label)
    if parent is None:
        # Fall back to table plane; relative object not spawned yet.
        return generators["Table"], _region_from_lims(xlim, ylim), _normalize_place_tag(extra_tag)
    parent_pos = parent.get("default_pos") or [0.0, 0.0, 0.0]
    parent_ori = parent.get("default_ori") or [1.0, 0.0, 0.0, 0.0]
    gen = ClutteredGenerator()
    # xlim/ylim are parent-local. The frame applies the full parent pose,
    # so a rotated cup does not spin samples around the world origin.
    gen.reset(
        box(-5.0, -5.0, 5.0, 5.0),
        frame=np.array(
            [
                float(parent_pos[0]),
                float(parent_pos[1]),
                float(parent_pos[2]),
                float(parent_ori[0]),
                float(parent_ori[1]),
                float(parent_ori[2]),
                float(parent_ori[3]),
            ],
            dtype=float,
        ),
    )
    return gen, _region_from_lims(xlim, ylim), _normalize_place_tag(extra_tag)


def _build_instance_record(
    *,
    object_type: str,
    category: str,
    category_idx: int,
    label: str | None,
    common: dict[str, Any],
    pose: np.ndarray,
    metadata: dict[str, Any] | None,
    cluttered: bool,
    yaml_path: str | None = None,
    relative_plane: str | None = None,
) -> dict[str, Any]:
    physics = deepcopy((metadata or {}).get("physics") or {})
    physics["type"] = PHYSICS_TYPE.get(object_type, "rigid")
    if "friction" not in physics and not cluttered:
        physics["friction"] = 0.3
    visual = deepcopy((metadata or {}).get("visual") or {})
    pose = np.asarray(pose, dtype=float).reshape(-1)
    record: dict[str, Any] = {
        "category": category,
        "category_idx": int(category_idx),
        "group": common.get("group"),
        "xlim": _to_plain(common.get("xlim")),
        "ylim": _to_plain(common.get("ylim")),
        "zlim": _to_plain(common.get("zlim")),
        "qpos": _to_plain(common.get("qpos") or [1, 0, 0, 0]),
        "rotate_deg": _to_plain(common.get("rotate_deg", 0)),
        "rotate_rand": bool(common.get("rotate_rand", False)),
        "relative_plane": relative_plane if relative_plane is not None else common.get("relative_plane", "Table"),
        "place_tag": common.get("place_tag"),
        "margin": float(common.get("margin", 0.01)),
        "check_mode": common.get("check_mode", "bbox"),
        "need_check_stable": bool(common.get("need_check_stable", True)),
        "label": label,
        "default_pos": [float(pose[0]), float(pose[1]), float(pose[2])],
        "default_ori": [float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])],
        "scale": _to_plain(common.get("scale") or [1.0, 1.0, 1.0]),
        "physics": _to_plain(physics),
        "visual": _to_plain(visual) if visual is not None else {},
    }
    if cluttered:
        record = {
            "category_idx": int(category_idx),
            "physics": {"type": "rigid"},
            "scale": [1.0, 1.0, 1.0],
            "type": "cluttered",
            "clutter_idx": 0,
            "yaml_path": yaml_path or "Clutter/clutter.yml",
            "relative_plane": relative_plane or "Table",
            "default_pos": record["default_pos"],
            "default_ori": record["default_ori"],
        }
    return record


def _place_one(
    *,
    object_type: str,
    category: str,
    category_idx: int,
    label: str | None,
    common: dict[str, Any],
    generators: dict[str, ClutteredGenerator],
    table_info: dict[str, Any],
    ground_info: dict[str, Any],
    placed: dict[str, dict[str, Any]],
    used_parents: set[str],
    cluttered: bool,
    yaml_path: str | None = None,
    relative_override: str | None = None,
) -> dict[str, Any]:
    modeldir = _object_dir(object_type, category, cluttered=cluttered)
    metadata = load_object_metadata(modeldir, category_idx)
    if metadata is None:
        raise UnStableError(f"missing metadata for {object_type}/{category}/{category_idx:05d}")
    plane = relative_override or _pick_relative_plane(common, used_parents)
    parent_label, support_key = _split_relative_plane(plane)
    parent = None if parent_label in ("Table", "Ground") else placed.get(parent_label)
    support_local = _support_local_pose(parent, support_key)
    if parent is not None and support_local is not None:
        pose = _pose_on_support(parent, support_local, metadata, common)
        reserved = _reserve_support_footprint(
            generators["Table"],
            pose,
            metadata,
            common,
            name=label or f"{category}_{category_idx}",
            parent_label=parent_label,
        )
        if not reserved:
            raise UnStableError(f"failed to place {category}/{category_idx} label={label}")
        return _build_instance_record(
            object_type=object_type,
            category=category,
            category_idx=category_idx,
            label=label,
            common=common,
            pose=pose,
            metadata=metadata,
            cluttered=cluttered,
            yaml_path=yaml_path,
            relative_plane=plane,
        )
    gen, allowed, extra_place = _generator_for_plane(
        plane,
        generators,
        placed,
        table_info,
        ground_info,
        common.get("xlim"),
        common.get("ylim"),
    )
    place_tag = _normalize_place_tag(common.get("place_tag"))
    if extra_place:
        place_tag = list(dict.fromkeys((place_tag or []) + extra_place))
    ok, pose, _poly = gen.add_model_from_config(
        config=metadata,
        place_tag=place_tag,
        rotate_rand=bool(common.get("rotate_rand", False)),
        rotate_deg=common.get("rotate_deg"),
        margin=float(common.get("margin", 0.01 if not cluttered else 0.015)),
        name=label or f"{category}_{category_idx}",
        allowed_region=allowed,
        zlim=common.get("zlim"),
        qpos=common.get("qpos"),
        cluttered=cluttered,
        check_mode=str(common.get("check_mode", "bbox")),
    )
    if not ok or pose is None:
        raise UnStableError(f"failed to place {category}/{category_idx} label={label}")
    return _build_instance_record(
        object_type=object_type,
        category=category,
        category_idx=category_idx,
        label=label,
        common=common,
        pose=pose,
        metadata=metadata,
        cluttered=cluttered,
        yaml_path=yaml_path,
        relative_plane=plane,
    )


def _place_choices(
    choices: list[tuple[str, int, str, dict[str, Any]]],
    *,
    object_type: str,
    generators: dict[str, ClutteredGenerator],
    table_info: dict[str, Any],
    ground_info: dict[str, Any],
    placed: dict[str, dict[str, Any]],
    used_parents: set[str],
    cluttered: bool,
    yaml_path: str | None,
) -> list[dict[str, Any]]:
    instances = []
    for category, category_idx, label, common in choices:
        inst = _place_one(
            object_type=object_type,
            category=category,
            category_idx=category_idx,
            label=None if cluttered else label,
            common=common,
            generators=generators,
            table_info=table_info,
            ground_info=ground_info,
            placed=placed,
            used_parents=used_parents,
            cluttered=cluttered,
            yaml_path=yaml_path,
        )
        instances.append(inst)
    return instances


def _sample_group(
    *,
    attempts: list[list[tuple[str, int, str, dict[str, Any]]]],
    object_type: str,
    generators: dict[str, ClutteredGenerator],
    table_info: dict[str, Any],
    ground_info: dict[str, Any],
    placed: dict[str, dict[str, Any]],
    used_parents: set[str],
    cluttered: bool,
    yaml_path: str | None = None,
) -> list[dict[str, Any]]:
    last_error: UnStableError | None = None
    for attempt_index, choices in enumerate(attempts):
        # A multi-object attempt that fails midway has already reserved footprints.
        # Only single-object variant groups (the mallet stand) are retried.
        if attempt_index > 0 and len(choices) != 1:
            break
        try:
            return _place_choices(
                choices,
                object_type=object_type,
                generators=generators,
                table_info=table_info,
                ground_info=ground_info,
                placed=placed,
                used_parents=used_parents,
                cluttered=cluttered,
                yaml_path=yaml_path,
            )
        except UnStableError as exc:
            last_error = exc
            if len(choices) != 1:
                raise
    if last_error is None:
        raise UnStableError("no placement attempt")
    raise last_error


def _resolve_clutter_yaml(yaml_path: str | None) -> str:
    path = yaml_path or "Clutter/clutter.yml"
    if os.path.isabs(path):
        return path
    candidate = os.path.join(OBJECTS_PATH, path)
    if os.path.isfile(candidate):
        return candidate
    return os.path.join(OBJECTS_PATH, "Clutter", os.path.basename(path))


def _clutter_pool(yaml_path: str) -> list[tuple[str, int]]:
    data = load_yaml(yaml_path)
    clutter = data.get("Clutter") or {}
    pool = []
    for name, indices in clutter.items():
        for idx in indices or []:
            pool.append((str(name), int(idx)))
    return pool


def _sample_clutter(
    *,
    layout: dict[str, Any],
    clutter_groups: list[Any],
    generators: dict[str, ClutteredGenerator],
    table_info: dict[str, Any],
    ground_info: dict[str, Any],
    placed: dict[str, dict[str, Any]],
    used_parents: set[str],
) -> None:
    if not clutter_groups:
        return
    rigid_bucket = layout.setdefault("Rigid", {})
    for group_cfg in clutter_groups:
        if not isinstance(group_cfg, dict):
            continue
        yaml_path = group_cfg.get("yaml_path", "Clutter/clutter.yml")
        resolved = _resolve_clutter_yaml(yaml_path)
        pool = _clutter_pool(resolved)
        nums = _as_int(group_cfg.get("nums", 0), 0)
        if nums <= 0 or not pool:
            continue
        mode = str(group_cfg.get("mode", "allow_duplicate"))
        try:
            picks = _select_instances(pool, nums, mode)
        except UnStableError:
            # Fewer unique clutter cats than requested: fall back to with-replacement.
            picks = _select_instances(pool, nums, "allow_duplicate")
        common = {
            "xlim": group_cfg.get("xlim"),
            "ylim": group_cfg.get("ylim"),
            "rotate_rand": group_cfg.get("rotate_rand", True),
            "rotate_deg": group_cfg.get("rotate_deg", 30),
            "relative_plane": group_cfg.get("relative_plane", "Table"),
            "margin": group_cfg.get("margin", 0.015),
            "check_mode": group_cfg.get("check_mode", "bbox"),
        }
        for category, category_idx in picks:
            try:
                inst = _place_one(
                    object_type="Rigid",
                    category=category,
                    category_idx=category_idx,
                    label=None,
                    common=common,
                    generators=generators,
                    table_info=table_info,
                    ground_info=ground_info,
                    placed=placed,
                    used_parents=used_parents,
                    cluttered=True,
                    yaml_path=yaml_path,
                    relative_override=str(common.get("relative_plane") or "Table"),
                )
            except UnStableError:
                continue
            rigid_bucket.setdefault(category, []).append(inst)


def _strip_random(cfg: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(cfg)
    out.pop("random", None)
    materials = out.get("materials")
    if isinstance(materials, dict):
        materials["random"] = False
    return out


def _fixture_room(scene_config: dict[str, Any]) -> dict[str, Any]:
    room = _strip_random(scene_config.get("Room") or {})
    if "default_ori" in room and "default_rot" not in room:
        room["default_rot"] = list(room["default_ori"])
    return room


def _fixture_table(scene_config: dict[str, Any]) -> dict[str, Any]:
    table = _strip_random(scene_config.get("Table") or {})
    table.pop("random", None)
    return table


def _fixture_ground(scene_config: dict[str, Any]) -> dict[str, Any]:
    return _strip_random(scene_config.get("Ground") or {})


def _fixture_background(scene_config: dict[str, Any]) -> dict[str, Any]:
    bg = _strip_random(scene_config.get("Background") or {})
    default = bg.pop("default", "brown_photostudio_02_4k.hdr")
    bg.pop("random", None)
    bg["category_name"] = default
    return bg
