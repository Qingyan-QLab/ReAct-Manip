"""Physics-based button / indicator matching task.

Buttons are MuJoCo slide joints with spring-damper return forces.  Indicator
state is derived from the measured button displacement; no qpos/qvel is
teleported during a step. A correct press scores twenty points and samples a new
target instruction; an incorrect press or timeout deducts twenty points.
The final signed sum is clamped to [0, 100] when the episode ends.
"""
from __future__ import annotations

import tempfile
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from react_manip.core import SuccessRule, TaskSpec
from react_manip.core.visual_randomization import apply_visual_scenario, resolve_scenario
from react_manip.visual_task import ManipulationTask

ROOT = Path(__file__).resolve().parent

SPEC = TaskSpec(
    task_id="button_indicator_matching", task_version="1.5", task_type="event",
    required_arms=1, required_gripper="parallel", required_eef_dof=6,
    supported_robots=("dual_piperx", "panda", "sawyer", "piperx", "mobile_dual_piperx"),
    horizon_sec=120.0, control_hz=20.0, done_behavior="stop",
    success=SuccessRule(
        min_occupancy=0.0, min_events=0, max_failures=None,
        require_terminal_success=True,
    ),
    metadata={"score_min": 0.0, "score_max": 100.0},
)

LABEL_MODES = ("numeric", "letters", "colors")
LIGHT_COLORS = {
    "red": (0.95, 0.05, 0.04), "yellow": (0.95, 0.78, 0.04),
    "blue": (0.05, 0.25, 0.95), "green": (0.05, 0.85, 0.18),
    "purple": (0.60, 0.12, 0.90), "orange": (0.98, 0.35, 0.03),
}
_COLOR_NAMES = tuple(LIGHT_COLORS)
CORRECT_PRESS_SCORE = 20.0
SUCCESS_SCORE_THRESHOLD = 100.0
TABLE_TOP_Z = 0.780
BUTTON_TRAVEL_M = 0.012
BUTTON_CLEARANCE_M = 0.0005
BUTTON_HOLD_REQUIRED_SEC = 0.5

# Task-level appearance variants.  They affect only render geometry/materials;
# button slide joints, collision groups, and indicator state remain physical.
BUTTON_STYLES = {
    "round": ("cylinder", (0.035, 0.008)),
    "mushroom": ("cylinder", (0.042, 0.010)),
    "square": ("box", (0.038, 0.038, 0.008)),
    "dome": ("sphere", (0.038,)),
}
BUTTON_MATERIALS = {
    "black_rubber": (".08 .09 .10 1", "0.05", "0.0"),
    "blue_polymer": (".05 .20 .58 1", "0.12", "0.0"),
    "red_polymer": (".72 .08 .06 1", "0.10", "0.0"),
    "yellow_polymer": (".90 .58 .03 1", "0.10", "0.0"),
    "ivory_plastic": (".78 .76 .68 1", "0.20", "0.0"),
    "brushed_steel": (".46 .49 .53 1", "0.55", "0.15"),
}
INDICATOR_STYLES = {
    "beacon": ("sphere", (0.028,), "sphere", (0.018,), 0.016),
    "ring": ("cylinder", (0.030, 0.010), "cylinder", (0.020, 0.003), 0.014),
    "square_panel": ("box", (0.030, 0.030, 0.010), "box", (0.020, 0.020, 0.003), 0.014),
    "dome": ("sphere", (0.034,), "sphere", (0.021,), 0.018),
}
INDICATOR_MATERIALS = {
    "black_housing": (".025 .028 .032 1", "0.65"),
    "white_housing": (".72 .74 .76 1", "0.35"),
    "steel_housing": (".30 .33 .37 1", "0.70"),
    "blue_housing": (".03 .08 .18 1", "0.50"),
}


def _size_text(values):
    return " ".join(f"{float(value):.6g}" for value in values)


def _rgba_values(raw):
    return np.asarray([float(value) for value in str(raw).split()], dtype=float)


def _styled_geom(parent, *, name, geom_type, size, **attrs):
    attrs = dict(attrs)
    attrs.update(name=name, type=geom_type, size=_size_text(size))
    return ET.SubElement(parent, "geom", **attrs)


def _vertical_half_extent(geom_type, size):
    """Return local Z half-extent for the supported axis-aligned caps."""
    if geom_type == "box":
        return float(size[2])
    if geom_type == "cylinder":
        return float(size[1])
    if geom_type == "sphere":
        return float(size[0])
    raise ValueError(f"unsupported button geom type: {geom_type}")


def _xml_scene(n: int, labels: list[str], colors: list[tuple[float, float, float]], *,
               button_style="round", button_material="black_rubber",
               indicator_style="beacon", indicator_material="black_housing",
               layout_seed=0, resolved_layout=None) -> str:
    button_geom_type, button_size = BUTTON_STYLES.get(button_style, BUTTON_STYLES["round"])
    button_half_height = _vertical_half_extent(button_geom_type, button_size)
    # At maximum depression the collision cap may touch but never penetrate
    # the tabletop. Previously every style used z=0.788; the 38 mm spherical
    # dome therefore started about 30 mm inside the table and was effectively
    # locked by contact forces (most visible for scenario seed 15).
    button_rest_z = TABLE_TOP_Z + BUTTON_TRAVEL_M + button_half_height + BUTTON_CLEARANCE_M
    button_rgba, button_specular, button_reflectance = BUTTON_MATERIALS.get(button_material, BUTTON_MATERIALS["black_rubber"])
    housing_type, housing_size, lens_type, lens_size, lens_offset = INDICATOR_STYLES.get(indicator_style, INDICATOR_STYLES["beacon"])
    housing_rgba, housing_specular = INDICATOR_MATERIALS.get(indicator_material, INDICATOR_MATERIALS["black_housing"])
    root = ET.Element("mujoco", model="button_indicator_matching")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "material", name="table_mat", rgba="0.48 0.30 0.16 1")
    ET.SubElement(asset, "material", name="button_mat", rgba=button_rgba,
                  specular=button_specular, reflectance=button_reflectance)
    ET.SubElement(asset, "material", name="lamp_off", rgba=housing_rgba,
                  specular=housing_specular, shininess="0.35")
    # The lens uses MuJoCo's emission term in addition to its runtime RGBA.
    # A separate halo and point light provide a soft local bloom / table spill
    # without making the whole room uniformly brighter.
    ET.SubElement(asset, "material", name="lamp_lens", rgba="1 1 1 1", emission="0.7", specular="0.35", shininess="0.45")
    ET.SubElement(asset, "material", name="lamp_halo", rgba="1 1 1 0.08", emission="0.35", specular="0", shininess="0")
    ET.SubElement(asset, "material", name="label_mat", rgba="0.02 0.02 0.02 1")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="room_light", pos="0 -1.5 2.5", dir="0 0 -1", directional="true", diffuse="0.65 0.65 0.65")
    # Robosuite's interactive renderer intentionally hides collision-only
    # geometry (group 0).  Keep a physical collision layer and an identical
    # visual layer for every task prop, otherwise the scene appears to contain
    # only the robot even though MuJoCo contacts are being simulated.
    ET.SubElement(world, "geom", name="floor", type="plane", pos="0 0 0", size="4 4 .01", rgba=".35 .35 .35 1", group="0", contype="1", conaffinity="1")
    ET.SubElement(world, "geom", name="floor_visual", type="plane", pos="0 0 0.001", size="4 4 .01", rgba="1 1 1 1", group="1", contype="0", conaffinity="0")
    table_attrs = dict(type="box", pos="0.25 0 0.755", size=".45 .80 .025", material="table_mat", friction=".5 .01 .001")
    ET.SubElement(world, "geom", name="table", group="0", contype="1", conaffinity="1", **table_attrs)
    ET.SubElement(world, "geom", name="table_visual", group="1", contype="0", conaffinity="0", rgba="1 1 1 1", **table_attrs)
    # Four slim legs make the work surface visibly grounded.  Each leg follows
    # the same collision/visual split as the tabletop; they are kept outside
    # the arm's nominal base footprint so they cannot create an initial contact.
    for leg_i, (x, y) in enumerate(((-0.12, -0.68), (-0.12, 0.68), (0.62, -0.68), (0.62, 0.68))):
        leg_attrs = dict(type="box", pos=f"{x:.3f} {y:.3f} 0.365", size=".04 .04 .365", material="table_mat", friction=".5 .01 .001")
        ET.SubElement(world, "geom", name=f"table_leg_{leg_i}", group="0", contype="1", conaffinity="1", **leg_attrs)
        ET.SubElement(world, "geom", name=f"table_leg_visual_{leg_i}", group="1", contype="0", conaffinity="0", rgba="1 1 1 1", **leg_attrs)
    # A compact seven-segment glyph makes numeric / letter labels visible in
    # rendered cameras without relying on unsupported MuJoCo text sites.
    segments = {
        "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg",
        "4": "bcfg", "5": "acdfg", "6": "acdefg",
        "A": "abcefg", "B": "cdefg", "C": "adef", "D": "bcdeg", "E": "adefg", "F": "aefg",
    }
    # Labels lie flat on the tabletop, just toward the robot from their lamp.
    # This makes "below the indicator" unambiguous in both the operator view
    # and the workspace top view, and avoids burying vertical glyphs inside
    # the light geometry.
    segment_offsets = {
        "a": (0.0, 0.014, "h"), "b": (0.012, 0.007, "v"), "c": (0.012, -0.007, "v"),
        "d": (0.0, -0.014, "h"), "e": (-0.012, -0.007, "v"), "f": (-0.012, 0.007, "v"),
        "g": (0.0, 0.0, "h"),
    }
    # Independent, stratified rows: retain safe spacing even with six wide
    # caps, without implying that visually aligned pairs share wiring.
    if resolved_layout is None:
        layout_rng = np.random.default_rng(layout_seed)
        row_y = np.linspace(-0.48, 0.48, n)
        button_xy = np.column_stack((layout_rng.uniform(-0.065, 0.025, n),
                                     row_y + layout_rng.uniform(-0.045, 0.045, n)))
        indicator_xy = np.column_stack((layout_rng.uniform(0.34, 0.46, n),
                                        row_y + layout_rng.uniform(-0.045, 0.045, n)))
    else:
        button_xy = np.asarray(resolved_layout['button_xy'], dtype=float)
        indicator_xy = np.asarray(resolved_layout['indicator_xy'], dtype=float)
        if any(a.shape != (n, 2) or not np.isfinite(a).all() for a in (button_xy, indicator_xy)):
            raise ValueError('resolved layout must contain finite N by 2 coordinates')
    for i, (button_x, y) in enumerate(button_xy):
        # Reserve the complete physical stroke above the solid tabletop. The
        # cap reaches the table only at its lower joint limit, independent of
        # whether it is a box, cylinder or sphere.
        body = ET.SubElement(world, "body", name=f"button_{i}", pos=f"{button_x:.6f} {y:.6f} {button_rest_z:.6f}")
        ET.SubElement(body, "joint", name=f"button_slide_{i}", type="slide", axis="0 0 1",
                      limited="true", range=f"-{BUTTON_TRAVEL_M:.6g} 0",
                      damping="2", stiffness="180", springref="0")
        # Keep the moving cap light enough that gravity does not look like a
        # press (a heavy cap would settle several millimetres against the
        # spring before any robot contact).  The slide range remains the hard
        # physical travel limit.
        button_attrs = dict(density="50", material="button_mat", rgba=button_rgba)
        _styled_geom(body, name=f"button_geom_{i}", geom_type=button_geom_type,
                     size=button_size, group="0", contype="1", conaffinity="1", **button_attrs)
        # Do not double the moving cap's mass/inertia with the render-only
        # duplicate.  The collision geom remains the sole physical mass.
        visual_button_attrs = dict(button_attrs)
        visual_button_attrs["density"] = "0"
        _styled_geom(body, name=f"button_visual_{i}", geom_type=button_geom_type,
                     size=button_size, group="1", contype="0", conaffinity="0", **visual_button_attrs)
        # A narrow render-only plunger fills the visible stroke below the cap.
        # It moves with the button but does not create a second collision body
        # that could jam against the tabletop at maximum depression.
        horizontal_extent = min(button_size[:2]) if button_geom_type == "box" else button_size[0]
        _styled_geom(
            body, name=f"button_stem_{i}", geom_type="cylinder",
            size=(min(0.014, float(horizontal_extent) * 0.38), BUTTON_TRAVEL_M / 2.0),
            pos=f"0 0 {-button_half_height - BUTTON_TRAVEL_M / 2.0:.6f}",
            group="1", contype="0", conaffinity="0", density="0",
            material="button_mat", rgba=button_rgba,
        )
        rgb = colors[i]
        # The indicator sphere rests just above the 0.78 m tabletop (radius
        # 28 mm), so it is visibly mounted rather than floating in space.
        indicator_x, y = indicator_xy[i]
        indicator_z = 0.81
        _styled_geom(world, name=f"indicator_{i}", geom_type=housing_type,
                     size=housing_size, pos=f"{indicator_x:.3f} {y:.6f} {indicator_z:.3f}",
                     material="lamp_off", rgba=housing_rgba,
                     group="1", contype="0", conaffinity="0")
        # A smaller raised lens and a translucent outer halo make the lamp
        # readable at camera distance while keeping the housing visibly dark.
        _styled_geom(world, name=f"indicator_lens_{i}", geom_type=lens_type,
                     size=lens_size, pos=f"{indicator_x:.3f} {y:.6f} {indicator_z + lens_offset:.3f}",
                     material="lamp_lens", rgba=".01 .01 .01 1",
                     group="1", contype="0", conaffinity="0")
        ET.SubElement(world, "geom", name=f"indicator_halo_{i}", type="sphere", pos=f"{indicator_x:.3f} {y:.6f} {indicator_z + 0.014:.3f}", size=".038", material="lamp_halo", rgba=".01 .01 .01 0.015", group="1", contype="0", conaffinity="0")
        # Task lights are preserved by shared visual randomization and are
        # switched on/off by _update_indicators from measured button state.
        # Indicator lights start disabled.  Their diffuse/specular terms are
        # enabled at runtime by ``_update_indicators`` only when the
        # corresponding button is physically pressed.  Keeping the XML
        # baseline dark makes hard-reset visual fingerprints stable while
        # preserving the task's dynamic illumination semantics.
        ET.SubElement(world, "light", name=f"indicator_light_{i}", pos=f"{indicator_x:.3f} {y:.6f} {indicator_z + 0.08:.3f}", dir="0 0 -1", directional="false", diffuse="0 0 0", specular="0 0 0", ambient="0 0 0", cutoff="55", exponent="2", castshadow="false")
        if labels[i] in segments:
            for segment in segments[labels[i]]:
                dx, dy, orientation = segment_offsets[segment]
                size = ".010 .002 .0015" if orientation == "h" else ".002 .010 .0015"
                label_x = indicator_x - 0.08
                label_y = y
                ET.SubElement(world, "geom", name=f"indicator_label_{i}_{labels[i]}_{segment}", type="box", pos=f"{label_x + dx:.6f} {label_y + dy:.6f} 0.7835", size=size, material="label_mat", group="1", contype="0", conaffinity="0")
        # Store the nominal color on a no-op site for deterministic XML data.
        ET.SubElement(world, "site", name=f"indicator_color_{i}", type="sphere", pos=f"{indicator_x:.3f} {y:.6f} 0.81", size=".001", rgba=f"{rgb[0]} {rgb[1]} {rgb[2]} 1", group="3")
    ET.SubElement(world, "camera", name="frontview", pos="0.95 -1.45 1.25", xyaxes="0.85 0.52 0 -0.27 0.44 0.85", fovy="52")
    root_xml = ET.tostring(root, encoding="unicode")
    return root_xml


class ButtonIndicatorArena(Arena):
    def __init__(self, n: int, labels: list[str], colors: list[tuple[float, float, float]], scenario=None, **appearance):
        # Arena currently accepts a file path rather than an XML string.  Use
        # a unique temporary path so parallel benchmark workers cannot parse
        # each other's scene while one worker is resetting.
        fd, raw_path = tempfile.mkstemp(prefix="cl-manip-button-indicator-", suffix=".xml")
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(_xml_scene(n, labels, colors, **appearance))
        try:
            super().__init__(raw_path)
        finally:
            Path(raw_path).unlink(missing_ok=True)
        apply_visual_scenario(self, scenario)


class ButtonIndicatorMatching(ManipulationEnv):
    """Press buttons to activate the randomly requested indicator."""
    step_handles_robot_action = True
    task_space_active_kp = 180.0
    task_space_passive_kp = 20.0

    @property
    def action_dim(self):
        return int(getattr(self, "_action_dim", 7))

    @property
    def total_score(self):
        """Keep signed progress; clamp only completed or truncated episodes."""
        if getattr(self, "_finished", False) or len(getattr(self, "_round_results", [])) == 5:
            return float(min(100.0, max(0.0, self._score)))
        return float(self._score)

    @property
    def elapsed_sec(self):
        return float(self.sim.data.time)

    @property
    def incorrect_press_penalty(self):
        """Each failed instruction deducts exactly 20 points."""
        return CORRECT_PRESS_SCORE

    def _score_is_success(self):
        return len(getattr(self, '_round_results', [])) == 5 and self._score == 100.0

    def __init__(self, robots="DualPiperX", *, seed=None, robot_id=None, scenario=None,
                 controller_configs=None, gripper_types="default", base_types="default",
                 initialization_noise=None,
                 has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                 horizon=2400, control_freq=20, camera_names=("frontview", "workspace_topview"),
                 camera_heights=256, camera_widths=256, camera_depths=False, **kwargs):
        self.scenario = resolve_scenario(scenario)
        self.robot_id = robot_id
        self._use_camera_obs = bool(use_camera_obs)
        self._camera_names = camera_names
        self.rng = np.random.default_rng(seed)
        params = self.scenario.parameters if self.scenario is not None else {}
        # Use a separate RNG so layout never changes wiring/appearance draws.
        # Saved scenarios reproduce layout independently of the caller's seed.
        self.layout_seed = params.get("layout", {}).get(
            "seed", self.scenario.seed if self.scenario is not None else seed
        )
        if self.layout_seed is None:
            self.layout_seed = int(np.random.SeedSequence().generate_state(1)[0])
        raw_count = params.get("button_count")
        self.n_buttons = int(raw_count) if raw_count is not None else int(self.rng.integers(3, 7))
        self.n_buttons = int(np.clip(self.n_buttons, 3, 6))
        raw_mode = params.get("label_mode")
        mode = str(raw_mode) if raw_mode is not None else str(self.rng.choice(LABEL_MODES))
        if mode not in LABEL_MODES: mode = "numeric"
        self.label_mode = mode
        if mode == "numeric": self.labels = [str(i + 1) for i in range(self.n_buttons)]
        elif mode == "letters": self.labels = [chr(ord("A") + i) for i in range(self.n_buttons)]
        else: self.labels = list(_COLOR_NAMES[: self.n_buttons])
        self.colors = [LIGHT_COLORS[name] for name in (self.labels if mode == "colors" else ["green"] * self.n_buttons)]
        # Appearance is sampled in the benchmark Profile.  The fallback RNG
        # keeps direct task construction varied as well, while preserving
        # exact reproducibility for a supplied seed.
        appearance = params.get("button", {}) if isinstance(params.get("button", {}), dict) else {}
        lamp_appearance = params.get("indicator", {}) if isinstance(params.get("indicator", {}), dict) else {}
        self.button_style = str(appearance.get("style", self.rng.choice(tuple(BUTTON_STYLES))))
        self.button_material = str(appearance.get("material", self.rng.choice(tuple(BUTTON_MATERIALS))))
        self.indicator_style = str(lamp_appearance.get("style", self.rng.choice(tuple(INDICATOR_STYLES))))
        self.indicator_material = str(lamp_appearance.get("material", self.rng.choice(tuple(INDICATOR_MATERIALS))))
        self._indicator_housing_rgba = _rgba_values(
            INDICATOR_MATERIALS.get(self.indicator_material, INDICATOR_MATERIALS["black_housing"])[0]
        )
        self.assignment = np.asarray(self.rng.permutation(self.n_buttons), dtype=int)
        # Explicit instance replay fixes wiring across episodes. Legacy calls
        # without this field retain their historical reset-time randomization.
        self._instance_assignment = params.get("button_indicator_assignment")
        if self._instance_assignment is not None:
            values = self._instance_assignment
            if (not isinstance(values, list) or any(type(x) is not int for x in values)
                    or sorted(values) != list(range(self.n_buttons))):
                raise ValueError("button_indicator_assignment must be a permutation")
            self._instance_assignment = np.array(values, dtype=int)
            self.assignment = self._instance_assignment.copy()
        self._episode_targets = None
        self._target_cursor = 0
        self._target_indicator = 0
        self._round_index = 0
        self._score = 0
        self._episode_index = -1
        self._events: list[str] = []
        self._prev_pressed = np.zeros(self.n_buttons, dtype=bool)
        self._press_started_sec = np.full(self.n_buttons, np.nan, dtype=float)
        self._correct_press_awarded = np.zeros(self.n_buttons, dtype=bool)
        if controller_configs is None:
            # Use the embodiment-specific controller file for the standard
            # single-arm models.  The generic BASIC composite contains left,
            # torso, head and base sections and otherwise emits harmless but
            # noisy "component missing" warnings for Panda/Sawyer.
            from react_manip.robots.resolution import default_controller_config
            controller_configs = default_controller_config(robots)
        super().__init__(robots=robots, controller_configs=controller_configs,
                         gripper_types=gripper_types, base_types=base_types,
                         has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer,
                         use_camera_obs=use_camera_obs, render_camera="frontview", camera_names=camera_names,
                         camera_heights=camera_heights, camera_widths=camera_widths, camera_depths=camera_depths,
                         control_freq=control_freq, horizon=horizon, hard_reset=True,
                         lite_physics=False, initialization_noise=initialization_noise, **kwargs)

    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos([-0.62, 0.0, 0.0])
        arena = ButtonIndicatorArena(
            self.n_buttons, self.labels, self.colors, self.scenario,
            button_style=self.button_style,
            button_material=self.button_material,
            indicator_style=self.indicator_style,
            indicator_material=self.indicator_material,
            layout_seed=self.layout_seed,
            resolved_layout=(self.scenario.parameters.get('layout')
                             if self.scenario is not None and 'button_xy' in self.scenario.parameters.get('layout', {}) else None),
        )
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots], mujoco_objects=[])

    def _setup_references(self):
        super()._setup_references()
        model = self.sim.model
        # During robosuite's first reference setup the composite controller
        # may not yet expose action_limits. Infer the stable layout from the
        # embodiment name instead of touching simulator state.
        self._action_dim = 14 if len(getattr(self.robots[0], "arms", ())) > 1 else 7
        self._button_qadr = np.asarray([int(model.jnt_qposadr[model.joint_name2id(f"button_slide_{i}")]) for i in range(self.n_buttons)], dtype=int)
        self._button_gids = [model.geom_name2id(f"button_geom_{i}") for i in range(self.n_buttons)]
        self._indicator_gids = [model.geom_name2id(f"indicator_{i}") for i in range(self.n_buttons)]
        self._indicator_lens_gids = [model.geom_name2id(f"indicator_lens_{i}") for i in range(self.n_buttons)]
        self._indicator_halo_gids = [model.geom_name2id(f"indicator_halo_{i}") for i in range(self.n_buttons)]
        self._indicator_light_ids = [model.light_name2id(f"indicator_light_{i}") for i in range(self.n_buttons)]

    def _eef_site_for_arm(self, arm="right"):
        sites = getattr(self.robots[0], "eef_site_id", {})
        if isinstance(sites, dict):
            if arm in sites:
                return int(sites[arm])
            return int(next(iter(sites.values())))
        return int(sites)

    def eef_pos(self, arm="right"):
        return np.asarray(self.sim.data.site_xpos[self._eef_site_for_arm(arm)], dtype=float).copy()

    def eef_quat(self, arm="right"):
        site = self._eef_site_for_arm(arm)
        matrix = np.asarray(self.sim.data.site_xmat[site], dtype=float).reshape(3, 3)
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, matrix.reshape(-1))
        return quat

    def eef_vel(self, arm="right"):
        site = self._eef_site_for_arm(arm)
        # site_xvelp is populated by MuJoCo's forward kinematics and avoids
        # finite-difference noise in geometry logs.
        values = getattr(self.sim.data, "site_xvelp", None)
        if values is None:
            return np.zeros(3, dtype=float)
        return np.asarray(values[site], dtype=float).copy()

    def _pressed(self):
        return np.asarray(self.sim.data.qpos[self._button_qadr] < -0.003, dtype=bool)

    def _sample_target(self, previous=None):
        if self._episode_targets is not None:
            if self._target_cursor >= len(self._episode_targets):
                raise ValueError("episode round plan exhausted; evaluation is invalid")
            target = self._episode_targets[self._target_cursor]
            self._target_cursor += 1
            return target
        if previous is None or self.n_buttons <= 1:
            return int(self.rng.integers(self.n_buttons))
        candidates = np.arange(self.n_buttons, dtype=int)
        candidates = candidates[candidates != int(previous)]
        return int(candidates[int(self.rng.integers(len(candidates)))])

    def _update_indicators(self):
        pressed = self._pressed()
        model = self.sim.model._model
        indicator_on = np.zeros(self.n_buttons, dtype=bool)
        # ``assignment[b]`` is the indicator controlled by button ``b``.
        # Multiple buttons may be held, so this is deliberately a scatter
        # operation rather than indexing by the indicator number.
        indicator_on[self.assignment[pressed]] = True
        for i, gid in enumerate(self._indicator_gids):
            lens_gid = self._indicator_lens_gids[i]
            halo_gid = self._indicator_halo_gids[i]
            light_id = self._indicator_light_ids[i]
            if indicator_on[i]:
                rgb = self.colors[i]
                # Keep a dark specular housing around a saturated emissive
                # lens; the halo alpha is intentionally subtle to avoid a
                # billboard-like glow.
                model.geom_rgba[gid] = self._indicator_housing_rgba
                model.geom_rgba[lens_gid] = [*rgb, 1.0]
                model.geom_rgba[halo_gid] = [*rgb, 0.10]
                model.light_diffuse[light_id] = np.asarray(rgb, dtype=float) * 0.75
                model.light_specular[light_id] = np.asarray(rgb, dtype=float) * 0.25
            else:
                model.geom_rgba[gid] = self._indicator_housing_rgba
                model.geom_rgba[lens_gid] = [0.008, 0.009, 0.010, 1.0]
                model.geom_rgba[halo_gid] = [0.008, 0.009, 0.010, 0.01]
                model.light_diffuse[light_id] = [0.0, 0.0, 0.0]
                model.light_specular[light_id] = [0.0, 0.0, 0.0]
        return pressed, indicator_on

    def _round_instruction(self):
        if getattr(self, '_finished', False) or len(self._round_results) == 5:
            return ""
        target = self.labels[self._target_indicator]
        if self.label_mode == "colors":
            instruction = f"Press the button associated with the {target} indicator light."
        else:
            # Labels belong to indicators, while the button-to-indicator mapping is
            # intentionally random. Do not imply that the button has the same
            # label as its target indicator.
            instruction = f"Press the button associated with indicator {target}."
        return instruction

    def _task_instruction(self):
        return (f"There are {self.n_buttons} buttons and indicators with a fixed mapping for this task instance. "
                "Buttons are nearer the robot; indicators are on the farther side, randomly assigned. "
                "Complete exactly five instructions, each with 24 seconds from publication. "
                "Hold the correct button for at least 0.5 seconds to earn +20. "
                "A new incorrect press or instruction timeout earns -20 and ends that instruction. "
                "A wrong press takes priority over a correct hold on the same tick. "
                "Release held buttons before pressing for the next instruction. "
                "The running total is the signed sum; when the episode ends, clamp it to 0 through 100. "
                "Negative final totals become 0; all five correct is success. "
                f"The overall time limit is {self.horizon / self.control_freq:g} seconds.")

    def _prompt_text(self):
        return f"{self._round_instruction()} {self._task_instruction()}"

    def get_episode_language_prompt(self):
        target = self.labels[self._target_indicator]
        text = self._prompt_text()
        return {
            "task_id": SPEC.task_id, "instruction": text, "success_condition": "Answer all five instructions correctly; the episode ends after five results.",
            "task_instruction": self._task_instruction(),
            "round_instruction": self._round_instruction(),
            "round_index": self._round_index if self._round_instruction() else None,
            "robot_id": self.robot_id, "episode_index": max(0, self._episode_index), "scenario_id": getattr(self.scenario, "scenario_id", None),
            "constraints": [f"Hold a correct button for at least {BUTTON_HOLD_REQUIRED_SEC:g} seconds while its indicator is lit to earn +{CORRECT_PRESS_SCORE:g}; releasing sooner earns no points.", "Each instruction ends with +20 for correct, or -20 for a wrong press or 24-second timeout.", "A released button turns its associated indicator off."],
            "public_context": {"button_count": self.n_buttons,
                                "target": target if self._round_instruction() else None,
                                "completed_rounds": len(self._round_results),
                                "status": "ended" if not self._round_instruction() else "in_progress",
                                "correct_press_score": CORRECT_PRESS_SCORE,
                                "correct_press_hold_sec": BUTTON_HOLD_REQUIRED_SEC,
                                "incorrect_press_penalty": self.incorrect_press_penalty,
                                "success_score_threshold": SUCCESS_SCORE_THRESHOLD,
                                "button_style": self.button_style,
                                "button_material": self.button_material,
                                "indicator_style": self.indicator_style,
                                "indicator_material": self.indicator_material}, "text": text,
        }

    def _observation(self, base_observation=None):
        pressed, indicator_on = self._update_indicators()
        observation = {
            "button_positions": self.sim.data.qpos[self._button_qadr].copy().astype(np.float32),
            "button_pressed": pressed.astype(np.float32),
            "indicator_on": indicator_on.astype(np.float32),
            "target_indicator": np.array([self._target_indicator], dtype=np.int32),
            "score": np.array([self.total_score], dtype=np.float32),
            "time": np.array([float(self.sim.data.time)], dtype=np.float32),
            "time_remaining_sec": np.array([max(0.0, float(self.horizon) / float(self.control_freq) - float(self.sim.data.time))], dtype=np.float32),
        }
        # Preserve robosuite's proprioception and optional RGB observations
        # for RL callers.  The task-owned scalar fields above are additive;
        # the oracle permutation remains available only through the geometry
        # adapter, never in the default policy observation.
        if isinstance(base_observation, dict):
            merged = dict(base_observation)
            merged.update(observation)
            return merged
        return observation

    def reset(self, *, seed=None, **kwargs):
        self._episode_targets = None
        self._target_cursor = 0
        if seed is not None: self.rng = np.random.default_rng(seed)
        value = super().reset()
        self._episode_index += 1
        self._score = 0; self._events = []
        self._round_results = []; self._round_started_sec = float(self.sim.data.time)
        self._finished = False
        # Synchronise the edge detector with the physically reset joints.  In
        # normal MuJoCo resets these are all released, but reading the state is
        # safer when a custom initialization or a warm reset is used.
        self._prev_pressed = self._pressed()
        self._press_started_sec.fill(np.nan)
        self._correct_press_awarded.fill(False)
        # The physical layout is fixed for an episode, but the wiring between
        # buttons and lamps is re-sampled at each reset as required by the
        # task definition.  A seed supplied to reset makes this permutation
        # reproducible.
        self.assignment = (self._instance_assignment.copy() if self._instance_assignment is not None
                           else np.asarray(self.rng.permutation(self.n_buttons), dtype=int))
        self._target_indicator = self._sample_target()
        self._round_index = 0
        return self._observation(value), {"success_state": False, "events": [], "total_score": 0.0, "raw_score": 0.0}

    def set_episode_targets(self, targets):
        """Install a resolved round plan at reset, never in the middle of play."""
        if self.timestep != 0:
            raise ValueError("round plan can only be installed at episode reset")
        if (not isinstance(targets, list) or len(targets) != 5
                or any(type(t) is not int or not 0 <= t < self.n_buttons for t in targets)
                or any(a == b for a, b in zip(targets, targets[1:]))):
            raise ValueError("invalid episode target sequence")
        self._episode_targets = tuple(targets)
        self._target_cursor = 0
        self._target_indicator = self._sample_target()
        self._round_index = 0
        self._round_started_sec = float(self.sim.data.time)
        self._update_indicators()

    def reward(self, action=None):
        """Robosuite reward hook.

        Event scoring is performed immediately after physics in :meth:`step`,
        where the measured button displacement and rising edge are available.
        The base loop still requires a finite scalar reward implementation.
        """
        del action
        return 0.0

    def _process_round(self, before, pressed, started, now):
        """One physical control tick can settle at most one instruction."""
        rising = (before | pressed) & ~self._prev_pressed
        for b in np.flatnonzero(rising):
            self._press_started_sec[b] = started
            self._events.append('button_pressed')
        self._press_started_sec[~pressed] = np.nan
        wrong = [int(self.assignment[b]) for b in np.flatnonzero(rising)
                 if self.assignment[b] != self._target_indicator]
        target_button = int(np.flatnonzero(self.assignment == self._target_indicator)[0])
        held = (pressed[target_button] and np.isfinite(self._press_started_sec[target_button])
                and now-self._press_started_sec[target_button] >= .5-1e-9)
        reason = 'incorrect' if wrong else 'correct' if held else 'timeout' if now-self._round_started_sec >= 24.-1e-9 else None
        delta = 0.; correct = []; incorrect = []; result = None
        if reason is not None:
            delta = 20. if reason == 'correct' else -20.
            result = dict(round_index=len(self._round_results), target=self._target_indicator,
                          outcome=reason, score=delta, elapsed_sec=now-self._round_started_sec)
            self._round_results.append(result); self._score += delta
            self._events.append({'correct':'correct_indicator','incorrect':'incorrect_indicator','timeout':'instruction_timeout'}[reason])
            self._events.append('instruction_completed')
            if reason == 'correct': correct.append(self._target_indicator)
            if reason == 'incorrect': incorrect = wrong
            # Held buttons cannot answer a new instruction until released/repressed.
            self._press_started_sec.fill(np.nan)
            self._correct_press_awarded.fill(False)
            if len(self._round_results) < 5:
                self._target_indicator = self._sample_target(self._target_indicator)
                self._round_index = len(self._round_results)
                self._round_started_sec = now
        self._prev_pressed = pressed.copy()
        return delta, correct, incorrect, result

    def step(self, action):
        if self._finished:
            raise RuntimeError("Episode ended; reset required")
        if not np.isfinite(np.asarray(action,dtype=float)).all():
            raise ValueError("action must be finite")
        if np.asarray(action).shape != (self.action_dim,):
            raise ValueError(f"expected action shape {(self.action_dim,)}")
        self._events = []
        correct_indicators = []
        incorrect_indicators = []
        score_delta = 0.0
        # A short control step can press and release a light button between
        # two controller observations.  Preserve the physically measured
        # state at the beginning of the interval so a genuine contact is not
        # lost merely because the spring began returning before ``step``
        # returned.  No state is synthesized; both samples come from MuJoCo.
        pressed_before = self._pressed()
        step_start_time = float(self.sim.data.time)
        base_result = super().step(np.clip(np.asarray(action, dtype=float), -1.0, 1.0))
        # Robosuite 1.x returns (obs, reward, done, info), while newer wrappers
        # may expose Gymnasium's five-tuple.  Preserve the actual observation
        # dictionary in both cases; passing the whole transition into
        # ``_observation`` silently dropped all robot proprioception after the
        # first step.
        if len(base_result) == 5:
            base_observation, _base_reward, base_terminated, base_truncated, base_info = base_result
        else:
            base_observation, _base_reward, base_done, base_info = base_result
            base_terminated, base_truncated = bool(base_done), False
        pressed, _indicator_on = self._update_indicators()
        score_delta, correct_indicators, incorrect_indicators, round_result = self._process_round(
            pressed_before, pressed, step_start_time, float(self.sim.data.time))
        score_success = self._score_is_success()
        terminated = len(self._round_results) == 5
        truncated = bool((base_terminated or base_truncated) and not terminated)
        self._finished = terminated or truncated
        reward = float(score_delta)
        info = dict(base_info or {})
        info.update({
            "success_state": score_success,
            "events": list(self._events),
            "total_score": self.total_score,
            "raw_score": float(self._score),
            "target_indicator": self._target_indicator,
            "completed_indicators": list(correct_indicators),
            "incorrect_indicators": list(incorrect_indicators),
            "score_delta": float(score_delta),
            "elapsed_sec": float(self.sim.data.time),
            "episode_finished": self._finished,
            "completed_rounds": len(self._round_results),
            "round_result": round_result,
            "round_remaining_sec": max(0., 24.-(float(self.sim.data.time)-self._round_started_sec)) if not self._finished else 0.,
        })
        info["terminal_success"] = score_success
        return self._observation(base_observation), reward, terminated, truncated, info

    def close(self):
        super().close()
