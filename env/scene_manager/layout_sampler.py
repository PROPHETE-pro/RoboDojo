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
from shapely.geometry import box
from shapely.ops import unary_union

from env.global_configs import OBJECTS_PATH
from utils.cluttered_generator import ClutteredGenerator, UnStableError
from utils.load_file import load_object_metadata, load_yaml

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

    for object_type in OBJECT_SPAWN_ORDER:
        groups = task_config.get(object_type) or []
        if not groups:
            continue
        type_bucket: dict[str, list[dict[str, Any]]] = {}
        for group in groups:
            for inst in _sample_group(
                group=group,
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


def _half_height(metadata: dict[str, Any] | None) -> float:
    if not metadata:
        return 0.0
    physics = metadata.get("physics") or {}
    size = physics.get("size")
    if isinstance(size, (list, tuple)) and len(size) >= 3:
        return abs(float(size[2])) / 2.0
    geom = ((metadata.get("geometry") or {}).get("oriented_bbox") or {}).get("extents")
    if isinstance(geom, (list, tuple)) and len(geom) >= 3:
        return abs(float(geom[2])) / 2.0
    return 0.0


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


def _parent_surface_z(parent: dict[str, Any]) -> float:
    pos = parent.get("default_pos") or [0.0, 0.0, 0.0]
    return float(pos[2]) + _half_height({"physics": parent.get("physics") or {}})


def _translate_region(region, dx: float, dy: float):
    if region is None:
        return None
    return shapely_translate(region, dx, dy)


def shapely_translate(geom, dx: float, dy: float):
    try:
        from shapely.affinity import translate

        return translate(geom, xoff=dx, yoff=dy)
    except Exception:
        return geom


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
    local_region = _region_from_lims(xlim, ylim)
    world_region = _translate_region(local_region, float(parent_pos[0]), float(parent_pos[1]))
    surface_z = _parent_surface_z(parent)
    # Keep xy sampling in world; lift z onto the parent top surface.
    gen.reset(
        generators["Table"].global_container,
        frame=np.array(
            [0.0, 0.0, surface_z, float(parent_ori[0]), float(parent_ori[1]), float(parent_ori[2]), float(parent_ori[3])],
            dtype=float,
        ),
    )
    for name, poly in generators["Table"].prohibited_area:
        gen.add_prohibited_area(poly, name=name)
    return gen, world_region, _normalize_place_tag(extra_tag)


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


def _sample_group(
    *,
    group: dict[str, Any],
    object_type: str,
    generators: dict[str, ClutteredGenerator],
    table_info: dict[str, Any],
    ground_info: dict[str, Any],
    placed: dict[str, dict[str, Any]],
    used_parents: set[str],
    cluttered: bool,
    yaml_path: str | None = None,
) -> list[dict[str, Any]]:
    common = dict(group.get("common") or {})
    select = group.get("select_mode") or {}
    nums = _as_int(select.get("nums", 1), 1)
    mode = str(select.get("mode", "allow_duplicate"))
    labels = list(select.get("label") or [])
    pool = _category_pool(group, object_type, cluttered=cluttered)
    picks = _select_instances(pool, nums, mode)
    instances = []
    for i, (category, category_idx) in enumerate(picks):
        label = labels[i] if i < len(labels) else f"{category}{i}"
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
