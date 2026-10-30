"""Shared pre-compilation visuals, reconstructed on every hard reset.

Ambient and shadow-free fill lights approximate diffuse illumination; this
is not physically based global illumination or an area-light renderer.
"""
import math
import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from .scenarios import Scenario


def resolve_scenario(value):
    return Scenario.load(value) if isinstance(value, (str, os.PathLike)) else value


class _StandaloneArena:
    """Minimal Arena-compatible wrapper for repository-owned MJCF strings."""

    def __init__(self, xml: str):
        self.root = ET.fromstring(xml)
        self.asset = self.root.find("asset")
        if self.asset is None:
            self.asset = ET.Element("asset")
            self.root.insert(0, self.asset)
        self.worldbody = self.root.find("worldbody")
        if self.worldbody is None:
            raise ValueError("MJCF visual randomization requires a worldbody")


def ensure_workspace_camera(arena, *, name: str = "workspace_topview") -> None:
    """Ensure every compiled scene has a canonical overhead work-area view.

    The camera is fixed in world coordinates at approximately 1.5 m height and
    points straight down. It is intentionally a non-physical sensor: adding
    it never changes collision,
    contacts, dynamics or task geometry.  Tasks may add a more tightly framed
    camera under another name, but this stable name is guaranteed for RL and
    frontends.
    """
    from react_manip.cameras import ensure_scene_cameras
    ensure_scene_cameras(arena)
    worldbody = getattr(arena, "worldbody", None)
    if worldbody is None:
        raise ValueError("workspace camera requires an arena with worldbody")
    if any(node.get("name") == name for node in worldbody.iter("camera")):
        return
    ET.SubElement(
        worldbody, "camera", name=name, pos="0 0 1.5",
        xyaxes="1 0 0 0 1 0", fovy="75",
    )


def apply_visual_scenario_xml(xml: str, scenario) -> str:
    """Apply the shared room / material / lighting contract to plain MJCF.

    Robosuite tasks call :func:`apply_visual_scenario` while constructing an
    ``Arena``. Native MuJoCo tasks use this equivalent pre-compilation path so
    the same sampled visual parameters reach every task model.
    """

    arena = _StandaloneArena(xml)
    apply_visual_scenario(arena, scenario)
    return ET.tostring(arena.root, encoding="unicode")


# Names are deliberately semantic rather than tied to one task.  File-backed
# PNGs are preferred for the default sampling pools because they preserve the
# scale variation, grain, and small imperfections seen on real surfaces.
# Procedural entries remain as lightweight fallbacks for headless/minimal
# installations.  The fourth value controls texrepeat (spatial scale).
PALETTES = {
    # File-backed materials (MuJoCo currently consumes PNG Color maps).  The
    # robosuite images are vendored into this repository so sampling does not
    # depend on a site-package path at runtime.
    "realistic-tiles": ("file", "assets/materials/realistic/tiles074/Tiles074_2K_Color.png", "", "3 3"),
    "realistic-wood": ("file", "assets/materials/realistic/woodfloor049/WoodFloor049_2K_Color.png", "", "2 8"),
    "realistic-tile-wall": ("file", "assets/materials/realistic/tiles074/Tiles074_2K_Color.png", "", "2 2"),
    "realistic-wood-panel": ("file", "assets/materials/realistic/woodfloor049/WoodFloor049_2K_Color.png", "", "1 5"),
    "oak-wood-realistic": ("file", "assets/materials/realistic/common/oak-wood.png", "", "2 2"),
    "walnut-wood-realistic": ("file", "assets/materials/realistic/common/walnut-wood.png", "", "2 2"),
    "gray-woodgrain-realistic": ("file", "assets/materials/realistic/common/gray-woodgrain.png", "", "2 2"),
    "varnished-wood-realistic": ("file", "assets/materials/realistic/common/varnished-wood-panels.png", "", "1 2"),
    "ceramic-realistic": ("file", "assets/materials/realistic/common/ceramic.png", "", "2 2"),
    "concrete-realistic": ("file", "assets/materials/realistic/common/light-gray-plaster.png", "", "2 2"),
    "gray-carpet-realistic": ("file", "assets/materials/realistic/common/gray-felt.png", "", "3 3"),
    "white-brick-realistic": ("file", "assets/materials/realistic/common/white-bricks.png", "", "2 2"),
    "cream-plaster-realistic": ("file", "assets/materials/realistic/common/cream-plaster.png", "", "2 2"),
    "brushed-steel-realistic": ("file", "assets/materials/realistic/common/steel-brushed.png", "", "4 2"),
    "scratched-steel-realistic": ("file", "assets/materials/realistic/common/steel-scratched.png", "", "4 2"),
    # Floors: common tile, wood, stone-like, rubber, carpet and linoleum.
    # Legacy procedural entries are retained for compatibility.  Profiles use
    # only a small fraction of them as intentionally simple fallback surfaces.
    "light-gray-floor-tile": ("file", "assets/materials/realistic/common/light-gray-floor-tile.png", "", "4 4"),
    "wood-tiles": ("file", "assets/materials/realistic/common/wood-tiles.png", "", "2 2"),
    "light-wood-parquet": ("file", "assets/materials/realistic/common/oak-wood.png", "", "2 5"),
    "dark-wood-parquet": ("file", "assets/materials/realistic/common/varnished-wood-panels.png", "", "2 5"),
    "slate-stone": ("file", "assets/materials/realistic/common/ceramic.png", "", "3 3"),
    "sandstone": ("file", "assets/materials/realistic/common/cream-plaster.png", "", "3 3"),
    "terrazzo": ("checker", ".72 .69 .61", ".42 .43 .40", "12 12"),
    "rubber-mat": ("flat", ".07 .08 .08", ".07 .08 .08", "1 1"),
    "blue-carpet": ("checker", ".10 .18 .26", ".16 .27 .37", "18 18"),
    "cream-linoleum": ("gradient", ".82 .78 .68", ".62 .59 .52", "2 2"),
    # Walls: paint/plaster, concrete, tile, brick and wood accents.
    "white-plaster": ("file", "assets/materials/realistic/common/cream-plaster.png", "", "2 2"),
    "cool-white-paint": ("file", "assets/materials/realistic/common/light-gray-plaster.png", "", "2 2"),
    "gray-plaster": ("file", "assets/materials/realistic/common/gray-plaster.png", "", "2 2"),
    "industrial-concrete": ("file", "assets/materials/realistic/common/light-gray-plaster.png", "", "3 3"),
    "blue-plaster": ("flat", ".28 .46 .65", ".28 .46 .65", "1 1"),
    "sage-plaster": ("flat", ".47 .58 .49", ".47 .58 .49", "1 1"),
    "warm-plaster": ("flat", ".73 .57 .39", ".73 .57 .39", "1 1"),
    "sand-plaster": ("gradient", ".78 .69 .54", ".56 .45 .32", "2 3"),
    "muted-yellow-paint": ("flat", ".72 .65 .38", ".72 .65 .38", "1 1"),
    "soft-pink-paint": ("flat", ".68 .46 .47", ".68 .46 .47", "1 1"),
    "brick-accent": ("file", "assets/materials/realistic/common/white-bricks.png", "", "2 2"),
    # Backwards-compatible aliases used by the desk randomizer.
    "light-wood": ("file", "assets/materials/realistic/common/oak-wood.png", "", "2 2"),
    "dark-wood": ("file", "assets/materials/realistic/common/walnut-wood.png", "", "2 2"),
    "maple-wood": ("file", "assets/materials/realistic/common/oak-wood.png", "", "2 2"),
    "walnut-wood": ("file", "assets/materials/realistic/common/walnut-wood.png", "", "2 2"),
    "white-laminate": ("flat", ".76 .77 .74", ".76 .77 .74", "1 1"),
    "black-laminate": ("flat", ".08 .09 .10", ".08 .09 .10", "1 1"),
    "brushed-metal": ("file", "assets/materials/realistic/common/steel-brushed.png", "", "4 2"),
    # Drawer fronts / interiors use a separate namespace in Profiles so the
    # same sampled desk material can coexist with a contrasting cabinet.
    "drawer-light-metal": ("file", "assets/materials/realistic/common/metal.png", "", "4 2"),
    "drawer-dark-metal": ("file", "assets/materials/realistic/common/steel-scratched.png", "", "4 2"),
    "drawer-white-laminate": ("flat", ".72 .73 .70", ".72 .73 .70", "1 1"),
    "drawer-blue-metal": ("gradient", ".18 .31 .42", ".07 .13 .19", "16 3"),
    "drawer-walnut": ("file", "assets/materials/realistic/common/walnut-wood.png", "", "2 2"),
}
# key RGB, fill RGB, ambient RGB, light count, key casts shadow
RIGS = {
    "overcast": (".22 .25 .30", ".17 .19 .22", ".20 .22 .25", 4, False),
    "warm_indoor": (".34 .25 .16", ".19 .15 .11", ".17 .14 .11", 3, False),
    "daylight_side": (".48 .45 .38", ".12 .17 .23", ".12 .14 .17", 3, True),
    "cool_office": (".27 .32 .39", ".17 .21 .27", ".17 .19 .23", 4, False),
    "ceiling_pair": (".28 .28 .28", ".19 .19 .19", ".18 .18 .18", 3, False),
    "hard_sun": (".75 .68 .50", ".05 .07 .10", ".08 .08 .08", 3, True),
    "backlit": (".10 .14 .22", ".45 .38 .25", ".10 .12 .16", 3, True),
    "mixed_daylight": (".55 .50 .42", ".25 .15 .08", ".22 .20 .17", 4, True),
}


def apply_visual_scenario(arena, scenario):
    scenario = resolve_scenario(scenario)
    ensure_workspace_camera(arena)
    if scenario is None:
        return
    p = scenario.parameters
    # Profile fields are grouped under ``appearance`` / ``runtime``; the
    # sampler intentionally strips that grouping in the resolved Scenario,
    # yielding ``parameters.room`` and ``parameters.lighting``.  Older
    # hand-authored scenarios use the same short form, so accept both layouts
    # and guarantee that sampled values reach the compiled MuJoCo model.
    appearance = p.get("appearance", {}) or {}
    runtime = p.get("runtime", {}) or {}
    room = appearance.get("room", p.get("room", {})) or {}
    lighting = runtime.get("lighting", p.get("lighting", {})) or {}
    desk = appearance.get("desk", p.get("desk", {})) or {}
    drawer = appearance.get("drawer", p.get("drawer", {})) or {}
    slot = appearance.get("slot", p.get("slot", {})) or {}
    block = appearance.get("block", p.get("block", {})) or {}
    layout = lighting.get("layout", "overcast")
    if layout not in RIGS:
        raise ValueError(f"Unknown lighting layout: {layout}")
    scale = float(lighting.get("intensity_scale", 1))
    azimuth = float(lighting.get("azimuth_deg", 45))
    if not math.isfinite(scale) or not 0 < scale <= 2 or not math.isfinite(azimuth):
        raise ValueError("Invalid lighting intensity / azimuth")
    for parent in arena.worldbody.iter():
        for child in list(parent):
            # Task-local lights (for example button indicator point lights)
            # are part of the interaction and must survive room-light
            # randomization.  Only replace the shared room rig here.
            if (child.tag == "light" and not child.get("name", "").startswith("indicator_light_")) \
                    or child.get("name", "").startswith("room_ceiling"):
                parent.remove(child)
    visual = arena.root.find("visual")
    if visual is None:
        visual = ET.SubElement(arena.root, "visual")
    head = visual.find("headlight")
    if head is None:
        head = ET.SubElement(visual, "headlight")
    head.attrib.update(active="0", ambient="0 0 0", diffuse="0 0 0", specular="0 0 0")
    key, fill, ambient, count, shadow = RIGS[layout]
    def rgb(raw, factor=1):
        return " ".join(str(float(v) * scale * factor) for v in raw.split())
    for i in range(count):
        a = math.radians(azimuth + i * 360 / count)
        ET.SubElement(arena.worldbody, "light", name=f"visual_light_{i}",
                      pos=f"{3*math.cos(a)} {3*math.sin(a)} 4",
                      dir=f"{-math.cos(a)} {-math.sin(a)} -1", directional="true",
                      diffuse=rgb(key if i == 0 else fill),
                      ambient=rgb(ambient, 1.7 / count), specular="0 0 0",
                      castshadow="true" if i == 0 and shadow else "false")
    def material(name, choice):
        if choice not in PALETTES:
            raise ValueError(f"Unknown texture palette: {choice}")
        builtin, c1, c2, texrepeat = PALETTES[choice]
        attrs = dict(name=name+"_tex", type="2d")
        if builtin == "file":
            # MuJoCo resolves file assets relative to the generated XML's
            # compiler directory.  Use a normalized absolute path here so
            # this remains valid for both robosuite's temporary XML and
            # direct MjModel.from_xml_string callers.
            attrs["file"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", c1))
        else: attrs.update(builtin=builtin, rgb1=c1, rgb2=c2, width="256", height="256")
        ET.SubElement(arena.asset, "texture", **attrs)
        ET.SubElement(arena.asset, "material", name=name, texture=name+"_tex",
                      texrepeat=texrepeat, texuniform="true", specular="0", reflectance="0")
    material("sampled_floor", room.get("floor_texture", "light-gray-floor-tile"))
    material("sampled_wall", room.get("wall_texture", "white-plaster"))
    floors = [g for g in arena.worldbody.iter("geom") if g.get("name", "") in ("floor", "floor_visual", "floor_vis")]
    for g in floors:
        g.set("material", "sampled_floor"); g.set("rgba", "1 1 1 1")
    if floors and not any(g.get("group") == "1" for g in floors):
        g = deepcopy(floors[0]); g.set("name", "sampled_floor_visual")
        g.set("group", "1"); g.set("contype", "0"); g.set("conaffinity", "0")
        arena.worldbody.append(g)
    walls = [g for g in arena.worldbody.iter("geom") if g.get("name", "").startswith("room_wall")]
    if not walls:
        # Legacy enclosures are visual only; no new task collisions.
        for i, (pos, size) in enumerate((("0 -8 2", "8 .05 2"), ("0 8 2", "8 .05 2"), ("-8 0 2", ".05 8 2"), ("8 0 2", ".05 8 2"))):
            walls.append(ET.SubElement(arena.worldbody, "geom", name=f"room_wall_visual_{i}", type="box", pos=pos, size=size, group="1", contype="0", conaffinity="0"))
    for g in walls:
        g.set("material", "sampled_wall"); g.set("rgba", "1 1 1 1")
    desk_texture = desk.get("texture")
    if desk_texture:
        material("sampled_desk", desk_texture)
        # Tasks use wood_mat, table_mat or pedestal_mat depending on whether
        # the surface is a desk, a lab table, or a turntable pedestal.  This
        # is a visual-only replacement; all geom friction/contact attributes
        # remain untouched.
        desk_names = {"table", "table_top", "table_top_vis", "desk", "desk_top"}
        for g in arena.worldbody.iter("geom"):
            if g.get("material") in {"wood_mat", "table_mat", "pedestal_mat"} or g.get("name", "") in desk_names:
                g.set("material", "sampled_desk")
    drawer_texture = drawer.get("texture")
    if drawer_texture:
        material("sampled_drawer", drawer_texture)
        # Restrict this replacement to drawer parts.  In particular, do not
        # recolour conveyor rails or other task-specific metal components.
        for g in arena.worldbody.iter("geom"):
            if g.get("name", "").startswith("drawer_"):
                g.set("material", "sampled_drawer")
    slot_texture = slot.get("texture")
    if slot_texture:
        material("sampled_slot", slot_texture)
        for g in arena.worldbody.iter("geom"):
            name = g.get("name", "")
            if g.get("material") == "box_mat" or name.startswith("slot_"):
                g.set("material", "sampled_slot")
                g.set("rgba", "1 1 1 1")
    block_texture = block.get("texture")
    if block_texture:
        material("sampled_block", block_texture)
        for g in arena.worldbody.iter("geom"):
            if g.get("material") == "block_mat" or g.get("name", "") == "block_visual":
                g.set("material", "sampled_block")
                g.set("rgba", "1 1 1 1")
