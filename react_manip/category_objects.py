"""Shared category-object assets and collision proxies, independent of tasks."""
from pathlib import Path
from typing import Any
import json
import xml.etree.ElementTree as ET
import numpy as np

OBJECT_LINEAR_SCALE = 1.55
OBJECT_CATEGORY_NAMES = (
    "footwear",
    "drinkware",
    "electronics",
    "toys",
    "kitchen_tools",
    "office_supplies",
    "personal_care",
    "planters",
)
CATEGORY_VARIANTS = {
    category: tuple(range(10)) for category in OBJECT_CATEGORY_NAMES
}
CATEGORY_COLORS = (
    (0.78, 0.16, 0.10, 1.0),
    (0.10, 0.58, 0.22, 1.0),
    (0.12, 0.30, 0.82, 1.0),
    (0.66, 0.18, 0.72, 1.0),
    (0.84, 0.48, 0.08, 1.0),
    (0.10, 0.62, 0.66, 1.0),
    (0.76, 0.28, 0.48, 1.0),
    (0.38, 0.56, 0.18, 1.0),
)
# Keep the existing shared download cache and attribution paths intact.
ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets/category_drawer_sorting"
ASSET_MANIFEST = ASSET_ROOT / "manifest.json"


def _load_asset_manifest() -> dict[str, Any]:
    if not ASSET_MANIFEST.is_file():
        return {"categories": {}}
    return json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))


_EXTERNAL_ASSET_MANIFEST = _load_asset_manifest()


def _external_asset(category: int, variant: int) -> dict[str, Any] | None:
    category_name = OBJECT_CATEGORY_NAMES[category]
    entries = _EXTERNAL_ASSET_MANIFEST.get("categories", {}).get(category_name, ())
    if variant >= len(entries):
        return None
    asset_id = str(entries[variant])
    directory = ASSET_ROOT / "gso" / asset_id
    metadata_path = directory / "cl_manip_asset.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mesh = directory / str(metadata.get("visual_mesh", "meshes/model.obj"))
    texture = directory / str(metadata.get("texture", "materials/textures/texture.png"))
    if not mesh.is_file() or not texture.is_file():
        return None
    return {**metadata, "mesh_path": mesh, "texture_path": texture}


def _vec(values) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


def _variant_spec(category: int, variant: int) -> dict[str, Any]:
    """Return one of ten deterministic, graspable variants per category."""

    if not 0 <= category < len(OBJECT_CATEGORY_NAMES):
        raise ValueError(f"invalid category index: {category}")
    if not 0 <= variant < 10:
        raise ValueError(f"variant must be in [0, 9], got {variant}")
    scale = 0.86 + 0.028 * variant
    aspect = 0.88 + 0.06 * (variant % 3)
    height = 0.024 + 0.0015 * (variant % 5)
    rgba = np.asarray(CATEGORY_COLORS[category], dtype=float).copy()
    rgba[:3] *= 0.86 + 0.025 * (variant % 6)
    if category == 0:  # footwear fallback: elongated sole plus rounded toe
        long = 0.040 * scale
        width = 0.022 * aspect
        half_height = 0.018 + 0.001 * (variant % 4)
        toe = 0.019 + 0.0008 * (variant % 5)
        geoms = [
            ("box", (long, width, half_height), (-0.006, 0.0, 0.0), None),
            ("sphere", (toe,), (long - 0.004, 0.0, 0.003), None),
        ]
        extent = np.array([long + toe, max(width, toe), max(half_height, toe + 0.003)])
    elif category == 1:  # drinkware fallback: upright canister
        radius = 0.030 * scale
        half_height = 0.030 + 0.0015 * (variant % 4)
        geoms = [("cylinder", (radius, half_height), (0.0, 0.0, 0.0), None)]
        extent = np.array([radius, radius, half_height])
    elif category == 2:  # electronics fallback: flat device with raised control area
        long = 0.043 * scale
        width = 0.030 * aspect
        half_height = 0.014 + 0.001 * (variant % 4)
        geoms = [
            ("box", (long, width, half_height), (0.0, 0.0, 0.0), None),
            ("box", (long * 0.45, width * 0.55, 0.004), (0.006, 0.0, half_height + 0.004), None),
        ]
        extent = np.array([long, width, half_height + 0.008])
    elif category == 3:  # toys fallback: stable dumbbell with rounded end caps
        long = 0.040 * scale
        radius = 0.018 + 0.001 * (variant % 4)
        half_height = 0.023 + 0.001 * (variant % 3)
        geoms = [
            ("box", (long, 0.013, half_height), (0.0, 0.0, 0.0), None),
            ("sphere", (radius,), (-long, 0.0, 0.006), None),
            ("sphere", (radius,), (long, 0.0, 0.006), None),
        ]
        extent = np.array([long + radius, radius, max(half_height, radius + 0.006)])
    elif category == 4:  # kitchen tools: elongated handle with a broad head
        long = 0.042 * scale
        width = 0.013 + 0.001 * (variant % 3)
        half_height = 0.012 + 0.001 * (variant % 4)
        geoms = [
            ("box", (long, width, half_height), (0.0, 0.0, 0.0), None),
            ("sphere", (width * 1.45,), (long, 0.0, 0.002), None),
        ]
        extent = np.array([long + width * 1.45, width * 1.45, max(half_height, width * 1.45)])
    elif category == 5:  # office supplies: compact trays, holders and cartridges
        long = 0.038 * scale
        width = 0.029 * aspect
        half_height = 0.020 + 0.0015 * (variant % 4)
        geoms = [
            ("box", (long, width, half_height), (0.0, 0.0, 0.0), None),
            ("box", (long * 0.72, width * 0.65, 0.004), (0.0, 0.0, half_height + 0.004), None),
        ]
        extent = np.array([long, width, half_height + 0.008])
    elif category == 6:  # personal care: bottle-like containers
        radius = 0.022 + 0.001 * (variant % 4)
        half_height = 0.034 * scale
        geoms = [
            ("cylinder", (radius, half_height), (0.0, 0.0, 0.0), None),
            ("cylinder", (radius * 0.52, 0.006), (0.0, 0.0, half_height + 0.006), None),
        ]
        extent = np.array([radius, radius, half_height + 0.012])
    else:  # planters: squat, stable containers
        radius = 0.031 * scale
        half_height = 0.027 + 0.0015 * (variant % 4)
        geoms = [
            ("cylinder", (radius, half_height), (0.0, 0.0, 0.0), None),
            ("cylinder", (radius * 1.08, 0.004), (0.0, 0.0, half_height + 0.004), None),
        ]
        extent = np.array([radius * 1.08, radius * 1.08, half_height + 0.008])
    # Normalize the horizontal grasp width, leaving clearance inside an
    # 8 cm parallel gripper. Uniform scaling preserves primitive geometry.
    scaled_extent = np.asarray(extent, dtype=float) * OBJECT_LINEAR_SCALE
    grasp_width = 2.0 * float(np.min(scaled_extent[:2]))
    grasp_scale = float(np.clip(grasp_width, 0.045, 0.060) / grasp_width)
    object_scale = OBJECT_LINEAR_SCALE * grasp_scale
    geoms = [
        (
            geom_type,
            tuple(np.asarray(size, dtype=float) * object_scale),
            tuple(np.asarray(offset, dtype=float) * object_scale),
            quat,
        )
        for geom_type, size, offset, quat in geoms
    ]
    return {
        "geoms": geoms,
        "extent": np.asarray(extent, dtype=float) * object_scale,
        "rgba": rgba,
    }


def _add_geom(parent, *, name: str, geom_type: str, size, pos, rgba, collision: bool):
    attrs = {
        "name": name,
        "type": geom_type,
        "size": _vec(size),
        "pos": _vec(pos),
        "rgba": _vec(rgba),
        "friction": "0.8 0.01 0.001",
        # 1 m drops reach ~4.4 m/s: the default 20 ms soft response can
        # penetrate a 5 cm tabletop. 5 ms remains above 2 physics steps.
        "solref": "0.005 1",
        # Include torsional and rolling resistance, not just sliding friction.
        "condim": "6" if collision else "3",
        "group": "0" if collision else "1",
        "contype": "1" if collision else "0",
        "conaffinity": "1" if collision else "0",
        "density": "320" if collision else "0",
    }
    return ET.SubElement(parent, "geom", **attrs)


def _add_object(asset, world, *, name: str, category: int, variant: int, pos) -> np.ndarray:
    spec = _variant_spec(category, variant)
    body = ET.SubElement(world, "body", name=name, pos=_vec(pos))
    ET.SubElement(body, "freejoint", name=f"{name}_free")
    external = _external_asset(category, variant)
    for index, (geom_type, size, offset, _quat) in enumerate(spec["geoms"]):
        _add_geom(
            body, name=f"{name}_collision_{index}", geom_type=geom_type,
            size=size, pos=offset, rgba=spec["rgba"], collision=True,
        )
        if external is None:
            _add_geom(
                body, name=f"{name}_visual_{index}", geom_type=geom_type,
                size=size, pos=offset, rgba=spec["rgba"], collision=False,
            )
    if spec["geoms"][0][0] == "cylinder":
        # A flat, inset foot supports upright containers with stable box/box
        # contacts. The analytic rim otherwise rocks on changing CCD points.
        # Extend only 1 mm below the rim; preserve the visible cylinder body.
        radius, half_height = spec["geoms"][0][1]
        foot_half_width = radius * 0.65
        _add_geom(
            body, name=f"{name}_collision_foot", geom_type="box",
            size=(foot_half_width, foot_half_width, 0.001),
            pos=(0, 0, -half_height), rgba=spec["rgba"], collision=True,
        )
    if external is not None:
        minimum = np.asarray(external["bounds_min"], dtype=float)
        maximum = np.asarray(external["bounds_max"], dtype=float)
        source_size = maximum - minimum
        target_size = 2.0 * np.asarray(spec["extent"], dtype=float) * 0.96
        # Fit each axis to the physical proxy. An inscribed uniform fit can
        # make a thin scan nearly invisible inside a much larger collider.
        mesh_scale = target_size / source_size
        source_center = (minimum + maximum) / 2.0
        mesh_name = f"{name}_scan_mesh"
        texture_name = f"{name}_scan_texture"
        material_name = f"{name}_scan_material"
        ET.SubElement(
            asset, "mesh", name=mesh_name,
            file=str(external["mesh_path"].resolve()),
            scale=_vec(mesh_scale),
        )
        ET.SubElement(
            asset, "texture", name=texture_name, type="2d",
            file=str(external["texture_path"].resolve()),
        )
        ET.SubElement(
            asset, "material", name=material_name, texture=texture_name,
            texuniform="false", specular="0.05", reflectance="0",
        )
        ET.SubElement(
            body, "geom", name=f"{name}_scan_visual", type="mesh",
            mesh=mesh_name, material=material_name,
            pos=_vec(-source_center * mesh_scale),
            group="1", contype="0", conaffinity="0", density="0",
        )
    return np.asarray(spec["extent"], dtype=float)
