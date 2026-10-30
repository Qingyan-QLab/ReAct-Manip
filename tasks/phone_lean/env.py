"""Interactive dual-arm phone / support scene."""
from __future__ import annotations
import math
import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from robosuite.models.tasks import ManipulationTask
from react_manip.robots.agilex_robosuite import controller_config, resolve_robot

ROOT = Path(__file__).resolve().parent
from .supports import LEGACY_SUPPORTS, SUPPORT_POOL, add_support
SUPPORT_TARGETS = LEGACY_SUPPORTS  # Historical scripts; runtime uses env.support_targets.
POINTS_PER_SUPPORT = 100.0
SCORE_MAX = 100.0
LEAN_STABLE_SECONDS = 10.0
LEAN_REQUIRED_FRACTION = 1.0
LEAN_MAX_DRIFT_M = 0.005
LEAN_MAX_LINEAR_SPEED_M_S = 0.005
LEAN_MAX_ANGULAR_SPEED_RAD_S = 0.05
LEAN_MIN_TABLE_ANGLE_DEG = 45.0
LEAN_MAX_TABLE_ANGLE_DEG = 75.0
CUP_MAX_TILT_DEG = 10.0

# Keep both PiPER-X grippers above the tabletop while the Episode is reset.
# The historical arm pose placed the right fingers directly in front of some
# valid phone layouts; the task controller moves to the phone only after the
# initial state has been accepted.
SAFE_ROBOT_ARM_QPOS = np.array([0.0, 1.5, -1.5, 0.0, 0.0, 0.0], dtype=float)


def check_initial_contacts(model, data, tolerance=1e-7, names=()):
    """Reject only invalid initial penetration; do not assess solvability."""
    body_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in names}
    body_ids.discard(-1)
    for contact in data.contact[:data.ncon]:
        first, second = int(contact.geom1), int(contact.geom2)
        body_first = int(model.geom_bodyid[first])
        body_second = int(model.geom_bodyid[second])
        if (body_first != body_second and (body_first in body_ids or body_second in body_ids)
                and contact.dist < -tolerance):
            geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
                          for index in (first, second)]
            raise ValueError(
                f"Initial penetration {contact.dist:.6g} m between {geom_names}")


def measure_lean(model, data, target):
    phone = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phone")
    position = data.xpos[phone]
    rotation = data.xmat[phone].reshape(3, 3)
    frame = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'phone_bottom_round')
    phone_scale = float(model.geom_size[frame, 0] / .0063)
    support_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target)
    if support_body < 0 or target == "phone":
        raise ValueError(f"Unknown support body {target!r}")
    back_support = front_support = bottom_table = gripper_contact = False
    for index, contact in enumerate(data.contact[:data.ncon]):
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        if body1 == phone:
            other = geom2
        elif body2 == phone:
            other = geom1
        else:
            continue
        force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, index, force)
        if force[0] < 1e-4:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
        local_contact = rotation.T @ (contact.pos - position)
        if int(model.geom_bodyid[other]) == support_body and local_contact[0] > 0.003*phone_scale:
            back_support = True
        if int(model.geom_bodyid[other]) == support_body and local_contact[0] < -0.003*phone_scale:
            front_support = True
        if name == "table" and local_contact[2] < -0.060*phone_scale:
            bottom_table = True
        if "gripper" in name:
            gripper_contact = True
    dof = int(model.jnt_dofadr[model.body_jntadr[phone]])
    table_angle = math.degrees(math.asin(float(np.clip(rotation[2, 2], -1, 1))))
    home_button_down = bool(rotation[2, 2] > 0.0)
    return {
        "screen_up": bool(-rotation[2, 0] >= -1e-10),
        "table_angle_deg": table_angle,
        "upright_lean": bool(
            LEAN_MIN_TABLE_ANGLE_DEG - 1e-10 <= table_angle <= LEAN_MAX_TABLE_ANGLE_DEG + 1e-10
        ),
        "back_support": back_support,
        "front_support": front_support,
        # The Home button is mounted on local -Z, so local +Z pointing upward
        # is exactly the required Home-button-down orientation.
        "home_button_down": home_button_down,
        "top_up": home_button_down,
        "bottom_table": bottom_table,
        "gripper_contact": gripper_contact,
        "linear_speed": float(np.linalg.norm(data.qvel[dof:dof + 3])),
        "angular_speed": float(np.linalg.norm(data.qvel[dof + 3:dof + 6])),
    }

def sample_is_valid(sample):
    return bool(
        sample["upright_lean"]
        and sample["home_button_down"]
        and sample["back_support"]
        and not sample["front_support"]
        and sample["bottom_table"]
        and not sample["gripper_contact"]
        and sample["linear_speed"] <= LEAN_MAX_LINEAR_SPEED_M_S
        and sample["angular_speed"] <= LEAN_MAX_ANGULAR_SPEED_RAD_S
    )




class PhoneLeanArena(Arena):
    """Scene with a free phone, tapered cup and physical supports."""
    def __init__(self, support_names=SUPPORT_POOL[:4]):
        super().__init__(str(ROOT / "assets" / "scene.xml"))
        texture_dir = ROOT.parent / "laptop_into_drawer" / "assets" / "textures"

        def add_texture(name, filename):
            ET.SubElement(self.asset, "texture", name=name, type="2d", file=str(texture_dir / filename))
            ET.SubElement(self.asset, "material", name=name + "_mat", texture=name,
                          texuniform="true", texrepeat="3 3", specular="0.25", shininess="0.35")

        add_texture("phonelean_table", "light-wood.png")
        ET.SubElement(self.asset, "material", name="phonelean_phone_back_mat",
                      rgba="0.48 0.75 0.92 1", specular="0.15", shininess="0.15")
        ET.SubElement(self.asset, "material", name="phonelean_phone_screen_mat",
                      rgba="0.012 0.015 0.020 1", specular="0.02", shininess="0.1")
        add_texture("phonelean_cup", "white-plaster.png")
        add_texture("phonelean_bowl", "white-plaster.png")
        add_texture("phonelean_support", "wood-tiles.png")

        table = self.worldbody.find("./geom[@name='table']")
        table.set("group", "1"); table.set("material", "phonelean_table_mat"); table.attrib.pop("rgba", None)
        phone = self.worldbody.find("./body[@name='phone']")
        phone.find("./geom[@name='phone_body']").set("group", "0")
        # The visible frame and rounded bottom define the outer shape.
        # Contact activation and zero-mass visual geometry are configured below.
        for name, pos, size, material in (
            ("phone_frame_visual", "0 0 0.00315", "0.0063 0.0385 0.07285", "phonelean_phone_back_mat"),
            ("phone_front_screen_visual", "-0.00615 0 0.003", "0.0002 0.0345 0.066", "phonelean_phone_screen_mat"),
        ):
            ET.SubElement(phone, "geom", name=name, type="box", pos=pos, size=size, group="1",
                          contype="0", conaffinity="0", material=material)
        rounded_visual = copy.deepcopy(phone.find("./geom[@name='phone_bottom_round']"))
        rounded_visual.set("name", "phone_bottom_round_visual")
        rounded_visual.set("group", "1")
        rounded_visual.set("mass", "0")
        rounded_visual.set("contype", "0"); rounded_visual.set("conaffinity", "0")
        rounded_visual.set("material", "phonelean_phone_back_mat")
        phone.append(rounded_visual)
        ET.SubElement(phone, "geom", name="phone_back_camera_visual", type="box",
                      pos="0.0062 -0.022 0.052", size="0.0002 0.008 0.010", group="1",
                      contype="0", conaffinity="0", rgba="0.03 0.03 0.035 1")
        # A flush Home button marks local -Z as the bottom. Its decorations
        # have no mass or collisions and cannot prop up the phone.
        for name, x, radius, color in (
            ("phone_home_button_ring", -.00639, .006, "0.78 0.84 0.90 1"),
            ("phone_home_button_center", -.00645, .0046, "0.012 0.015 0.020 1"),
        ):
            ET.SubElement(phone, "geom", name=name, type="cylinder",
                          pos=f"{x} 0 -0.052", size=f"{radius} 0.00003",
                          quat="0.7071067811865476 0 0.7071067811865476 0",
                          group="1", mass="0", contype="0", conaffinity="0", rgba=color)
        # No invisible grasp rails or skids. Each rendered phone primitive
        # below also serves as its own collision shape.

        # Remove the historical static support block. The ten-item support
        # catalog is built below and intentionally contains no cube.
        old_box = self.worldbody.find("./geom[@name='support_box']")
        if old_box is not None:
            self.worldbody.remove(old_box)

        # Remove the old static cylinder placeholders and put the cup in its
        # own free body. It starts on the table but can now be pushed over by
        # the gripper and settles under gravity/friction.
        for name in ("cup_lower", "cup_upper"):
            old = self.worldbody.find(f"./geom[@name='{name}']")
            if old is not None:
                self.worldbody.remove(old)
        cup_body = ET.SubElement(self.worldbody, "body", name="cup", pos="0.18 0 0.68")
        ET.SubElement(cup_body, "freejoint", name="cup_free")

        # MuJoCo's cone keyword is unavailable in the pinned binding. A
        # 64-sided thin-walled frustum gives both smooth visuals and a real
        # inner wall / rim for collision and friction.
        seg = 64; wall = 0.006; inner_bottom = 0.035 - wall; inner_top = 0.055 - wall
        cverts = []
        for radius, z in ((0.035, 0.0), (0.055, 0.14), (inner_bottom, 0.006), (inner_top, 0.14)):
            for i in range(seg):
                a = 2 * math.pi * i / seg; cverts.append((radius * math.cos(a), radius * math.sin(a), z))
        cfaces = []
        for i in range(seg):
            j = (i + 1) % seg
            # outer shell, inner shell (reverse winding), top rim, bottom ring
            cfaces += [[i, j, seg+j], [i, seg+j, seg+i]]
            cfaces += [[2*seg+i, 3*seg+j, 2*seg+j], [2*seg+i, 3*seg+i, 3*seg+j]]
            cfaces += [[seg+i, seg+j, 3*seg+j], [seg+i, 3*seg+j, 3*seg+i]]
            cfaces += [[i, 2*seg+i, 2*seg+j], [i, 2*seg+j, j]]
        mesh_vertices = " ".join(f"{v:.7f}" for p in cverts for v in p)
        mesh_faces = " ".join(str(v) for tri in cfaces for v in tri)
        ET.SubElement(self.asset, "mesh", name="phonelean_cup_visual_mesh",
                      vertex=mesh_vertices, face=mesh_faces)
        ET.SubElement(self.asset, "mesh", name="phonelean_cup_collision_mesh",
                      vertex=mesh_vertices, face=mesh_faces)
        # Decompose the hollow shell into 64 convex wedges using the exact
        # visual vertices. A single convex hull would incorrectly fill the cup.
        cup_friction = "0.8 0.003 0.0001"
        ET.SubElement(cup_body, "geom", name="phonelean_cup_bottom_collision", type="cylinder",
                      pos="0 0 0.005", size="0.035 0.004", mass="0.050",
                      friction=cup_friction, solref="0.006 1", solimp="0.92 0.98 0.002")
        rib_count = 64
        for i in range(rib_count):
            j = (i + 1) % rib_count
            vertices = [cverts[k*seg+t] for k in range(4) for t in (i,j)]
            mesh_name = f"phonelean_cup_segment_{i}"
            ET.SubElement(self.asset, "mesh", name=mesh_name,
                          vertex=" ".join(str(v) for xyz in vertices for v in xyz))
            ET.SubElement(cup_body, "geom", name=f"phonelean_cup_rib_{i}", type="mesh",
                          mesh=mesh_name, mass=str(.144/rib_count), friction=cup_friction,
                          solref="0.006 1", solimp="0.92 0.98 0.002")
        ET.SubElement(cup_body, "geom", name="phonelean_cup_visual", type="mesh",
                      mesh="phonelean_cup_visual_mesh", group="1",
                      contype="0", conaffinity="0", material="phonelean_cup_mat")
        ET.SubElement(cup_body, "geom", name="phonelean_cup_inner_bottom_visual", type="cylinder",
                      pos="0 0 0.006", size="0.029 0.001", group="1",
                      contype="0", conaffinity="0", material="phonelean_cup_mat")

        # A second open vessel: a shallow bowl with the same physical role as
        # the cup. Its visual mesh includes the inside and rim; collision uses
        # a stable bottom + tapered ribs + rim compound so it can be pushed or
        # tipped without concave-mesh instability.
        bowl_body = ET.SubElement(self.worldbody, "body", name="bowl", pos="0.38 0.20 0.68")
        ET.SubElement(bowl_body, "freejoint", name="bowl_free")
        bseg = 64; br0, br1, bh, bw = 0.055, 0.10, 0.075, 0.006
        biv = br0 - bw; bit = br1 - bw
        bverts = []
        for radius, z in ((br0, 0.0), (br1, bh), (biv, 0.005), (bit, bh)):
            for i in range(bseg):
                a = 2 * math.pi * i / bseg; bverts.append((radius * math.cos(a), radius * math.sin(a), z))
        bfaces = []
        for i in range(bseg):
            j = (i + 1) % bseg
            bfaces += [[i, j, bseg+j], [i, bseg+j, bseg+i]]
            bfaces += [[2*bseg+i, 3*bseg+j, 2*bseg+j], [2*bseg+i, 3*bseg+i, 3*bseg+j]]
            bfaces += [[bseg+i, bseg+j, 3*bseg+j], [bseg+i, 3*bseg+j, 3*bseg+i]]
            bfaces += [[i, 2*bseg+i, 2*bseg+j], [i, 2*bseg+j, j]]
        ET.SubElement(self.asset, "mesh", name="phonelean_bowl_visual_mesh",
                      vertex=" ".join(f"{v:.7f}" for p in bverts for v in p),
                      face=" ".join(str(v) for tri in bfaces for v in tri))
        bowl_friction = "0.90 0.50 0.07"
        ET.SubElement(bowl_body, "geom", name="bowl_bottom_collision", type="cylinder",
                      pos="0 0 0.004", size=f"{br0} 0.004", mass="0.045",
                      friction=bowl_friction, solref="0.006 1", solimp="0.92 0.98 0.002")
        for i in range(24):
            a = 2 * math.pi * i / 24; j = (i + 1) % 24
            ET.SubElement(bowl_body, "geom", name=f"bowl_rib_{i}", type="capsule",
                          fromto=f"{br0*math.cos(a):.7f} {br0*math.sin(a):.7f} 0.006 "
                                 f"{br1*math.cos(a):.7f} {br1*math.sin(a):.7f} {bh-0.006:.7f}",
                          size="0.005", mass="0.003", friction=bowl_friction,
                          solref="0.006 1", solimp="0.92 0.98 0.002")
            a2 = 2 * math.pi * j / 24
            ET.SubElement(bowl_body, "geom", name=f"bowl_rim_{i}", type="capsule",
                          fromto=f"{br1*math.cos(a):.7f} {br1*math.sin(a):.7f} {bh:.7f} "
                                 f"{br1*math.cos(a2):.7f} {br1*math.sin(a2):.7f} {bh:.7f}",
                          size="0.006", mass="0.0005", friction=bowl_friction,
                          solref="0.006 1", solimp="0.92 0.98 0.002")
        ET.SubElement(bowl_body, "geom", name="bowl_visual", type="mesh",
                      mesh="phonelean_bowl_visual_mesh", group="1", contype="0", conaffinity="0",
                      material="phonelean_bowl_mat")
        ET.SubElement(bowl_body, "geom", name="bowl_inner_bottom_visual", type="cylinder",
                      pos="0 0 0.005", size="0.049 0.001", group="1",
                      contype="0", conaffinity="0", material="phonelean_bowl_mat")
        # Explicit, moderately compliant sliding contact prevents the old
        # oversized torsional / rolling friction from rocking the free cup.
        contact = self.root.find('contact')
        if contact is None:
            contact = ET.SubElement(self.root, 'contact')
        ET.SubElement(contact,'pair',geom1='table',geom2='phonelean_cup_bottom_collision',
                      condim='3',friction='.8 .8 .003 .0001 .0001',
                      solref='.02 1',solimp='.9 .95 .001')
        # Four tiny physical pads under the base establish a stable support
        # polygon instead of cycling cylinder-plane contact points.
        for i,(x,y) in enumerate(((.022,.022),(.022,-.022),(-.022,.022),(-.022,-.022))):
            name=f'cup_base_pad_{i}'
            ET.SubElement(cup_body,'geom',name=name,type='sphere',pos=f'{x} {y} .002',
                          size='.002',mass='.0001',group='1',rgba='.16 .16 .16 1',
                          friction=cup_friction,solref='.02 1',solimp='.9 .95 .001')
            ET.SubElement(contact,'pair',geom1='table',geom2=name,condim='3',
                          friction='.8 .8 .003 .0001 .0001',solref='.02 1',solimp='.9 .95 .001')
        # Preserve the 180-g core inertia while using the actual frame,
        # panels and camera shapes for collision, without double-counting mass.
        body_geom = phone.find("./geom[@name='phone_body']")
        body_geom.set("contype", "0"); body_geom.set("conaffinity", "0")
        phone.find("./geom[@name='phone_bottom_round']").set("contype", "0")
        phone.find("./geom[@name='phone_bottom_round']").set("conaffinity", "0")
        for geom in phone.findall('geom'):
            if geom.get('name', '').endswith('_visual'):
                geom.set('contype', '1'); geom.set('conaffinity', '1')
                geom.set('mass', '0')
                geom.set('friction', '.65 .003 .0001')
                geom.set('solref', '.015 1')
        for geom in self.worldbody.iter("geom"):
            if geom.get("name", "").startswith(("support_box", "pyramid_rod")):
                geom.set("group", "1"); geom.set("material", "phonelean_support_mat"); geom.attrib.pop("rgba", None)

        ET.SubElement(self.worldbody, "geom", name="phonelean_robot_mount_plate", type="box",
                      pos="0 -0.535 0.678", size="0.075 0.055 0.002", group="1",
                      contype="0", conaffinity="0", material="phonelean_support_mat")
        ET.SubElement(self.worldbody, "camera", name="phonelean_overview",
                      pos="-1.20 -1.45 1.45", xyaxes="0.78 -0.63 0 0.33 0.41 0.85", fovy="58")
        ET.SubElement(self.worldbody, "camera", name="phonelean_topview",
                      pos="0 0 2.25", xyaxes="1 0 0 0 1 0", fovy="50")
        ET.SubElement(self.worldbody, "camera", name="phonelean_sideview",
                      pos="-1.35 0.15 1.10", xyaxes="0 -1 0 0.30 0 0.95", fovy="55")
        ET.SubElement(self.worldbody, "light", name="phonelean_key", pos="-0.7 -1.0 2.4",
                      dir="0.2 0.25 -1", diffuse="0.95 0.92 0.86", ambient="0.28 0.28 0.30", directional="false")
        ET.SubElement(self.worldbody, "light", name="phonelean_fill", pos="0.8 0.7 1.8",
                      dir="-0.25 -0.2 -1", diffuse="0.60 0.68 0.82", ambient="0.18 0.18 0.20", directional="false")

        # Build the historical assets once, then retain exactly the drawn set.
        selected = tuple(support_names)
        if len(selected) != 4 or len(set(selected)) != 4 or not set(selected) <= set(SUPPORT_POOL):
            raise ValueError("A scene requires four unique catalog supports")
        for name in LEGACY_SUPPORTS:
            if name not in selected:
                self.worldbody.remove(self.worldbody.find(f"./body[@name='{name}']"))
        if 'cup' not in selected:
            for pair in list(contact):
                contact.remove(pair)
        for name in selected:
            if name not in LEGACY_SUPPORTS:
                add_support(self.worldbody, name)


class PhoneLean(ManipulationEnv):
    """Free-body phone leaning with dual PiPER-X as the benchmark embodiment."""

    def __init__(self, robots="dual_piperx", controller_configs=None, gripper_types="default",
                 seed=None, task_seed=0, episode_seed=None, task_instance=None,
                 episode_instance=None, support_target=None, **kwargs):
        from .instances import load_profile, sample_task_instance, sample_episode_instance
        self.instance_profile = load_profile()
        self.task_seed = int(task_instance["task_seed"] if task_instance is not None else task_seed)
        self.episode_seed = int(episode_instance["episode_seed"] if episode_instance is not None
                                else (episode_seed if episode_seed is not None else (seed or 0)))
        self.layout_seed = self.episode_seed
        from .instances import validate_task_instance, validate_episode_instance
        self.task_instance = (sample_task_instance(self.task_seed, self.instance_profile)
                              if task_instance is None else copy.deepcopy(task_instance))
        validate_task_instance(self.task_instance, self.instance_profile)
        self.episode_instance = (sample_episode_instance(self.task_instance, self.episode_seed, self.instance_profile)
                                 if episode_instance is None else copy.deepcopy(episode_instance))
        validate_episode_instance(self.task_instance, self.episode_instance, self.instance_profile)
        noise = kwargs.pop("initialization_noise", None)
        if noise not in (None, 0):
            raise ValueError("Initial robot pose is fixed; use task_seed and episode_seed")
        self.support_targets = tuple(self.task_instance['support_names'])
        self.support_target = (self.support_targets[0] if support_target is None
                               else str(support_target))
        if self.support_target not in self.support_targets:
            raise ValueError(
                f"support_target must be one of {self.support_targets}, got {support_target!r}"
            )
        if robots not in ('piperx', 'dual_piperx', 'PiperX', 'DualPiperX', 'legacy_piperx', 'aloha', 'AgileXPiper', 'AgileXAloha', 'panda_libero', 'LiberoPanda'):
            raise ValueError('phone_lean supports dual_piperx and piperx')
        from react_manip.robots.resolution import resolve_robosuite_robot, robot_id_from_model
        if isinstance(robots,str) and robot_id_from_model(robots) in ('panda_libero','aloha_agilex_robotwin'):
            resolved=resolve_robosuite_robot(robot_id_from_model(robots))
            robots=resolved.robot_name
            controller_configs=resolved.controller_configs
            gripper_types=resolved.gripper_types
        self._is_aloha = robots in ('dual_piperx', 'DualPiperX', 'aloha', 'AgileXAloha')
        if robots in ('piperx', 'dual_piperx', 'legacy_piperx', 'aloha'):
            robots = resolve_robot(robots)
        if controller_configs is None:
            controller_configs = controller_config(('right', 'left') if self._is_aloha else ('right',))
            for config in controller_configs['body_parts'].values():
                config['kp'] = self.task_instance['simulation']['controller_kp']
        super().__init__(robots=robots, controller_configs=controller_configs,
                         gripper_types=gripper_types, has_renderer=kwargs.pop("has_renderer", False),
                         has_offscreen_renderer=kwargs.pop("has_offscreen_renderer", True),
                         use_camera_obs=kwargs.pop("use_camera_obs", False),
                         render_camera=kwargs.pop("render_camera", "phonelean_overview"),
                         camera_names=kwargs.pop("camera_names", "phonelean_overview"),
                         horizon=kwargs.pop("horizon", 3600), control_freq=20,
                         initialization_noise=None, seed=self.episode_seed, **kwargs)

    def _load_model(self):
        super()._load_model()
        from react_manip.robots.mounts import use_existing_mount
        use_existing_mount(self.robots[0].robot_model)
        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos([-0.3, -0.77375, 0.68] if self._is_aloha else [0.0, -0.535, 0. if robot_model.__class__.__name__=="LiberoPanda" else 0.68]); robot_model.set_base_ori(np.array([0.0, 0.0, math.pi / 2]))
        # Keep the central PiPER-X geometry; only task material/damping settings vary.
        for gripper in self.robots[0].gripper.values():
            # Critically damp the light sliding fingers before contact. This
            # is joint damping, not extra object/contact friction.
            for joint in gripper.worldbody.iter('joint'):
                joint.set('damping', str(self.task_instance['simulation']['finger_joint_damping']))
        # Higher priority preserves the exact fixed gripper contact parameters,
        # even when the contacted object's sampled friction is different.
        for gripper in self.robots[0].gripper.values():
            for geom in gripper.worldbody.iter("geom"):
                geom.set("priority", "100")
                geom.set("friction", " ".join(map(str, self.task_instance["gripper_contact_friction"])))
                geom.set("condim", str(self.task_instance["gripper_contact_condim"]))
        from .instances import apply_physics_xml
        arena = PhoneLeanArena(self.support_targets)
        from react_manip.cameras import ensure_scene_cameras, ensure_robot_cameras
        ensure_scene_cameras(arena, "phone_lean")
        ensure_robot_cameras(robot_model)
        apply_physics_xml(arena, self.task_instance)
        arena.set_origin([0.0, 0.0, 0.0])
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[robot_model], mujoco_objects=[])
        # ManipulationTask owns its option element; arena options are not merged.
        option = self.model.root.find("option")
        if option is None:
            option = ET.SubElement(self.model.root, "option")
        option.set("gravity", f"0 0 {-self.task_instance['gravity_m_s2']}")
        option.set("integrator", self.task_instance['simulation']['integrator'])
        option.set("tolerance", str(self.task_instance['simulation']['solver_tolerance']))
        flag = option.find('flag')
        if flag is None:
            flag = ET.SubElement(option, 'flag')
        # MuJoCo 3.3 defaults mesh/box pairs to one point. Flat fingers need
        # a contact patch, otherwise the phone rocks about the pinch axis.
        flag.set('multiccd', 'enable' if self.task_instance['simulation']['multiccd'] else 'disable')
        ET.SubElement(
            self.model.equality, "weld", name="phone_grasp_assist",
            body1="gripper0_right_eef", body2="phone", active="false",
            solref=".008 1", solimp=".95 .99 .001", torquescale="1",
        )

    def _setup_references(self):
        super()._setup_references()
        self.support_body_ids = {}
        for name in ('phone', *self.support_targets):
            body_id = self.sim.model.body_name2id(name)
            joint = self.sim.model.body_jntadr[body_id]
            setattr(self, name+'_body_id', body_id)
            setattr(self, name+'_qpos', int(self.sim.model.jnt_qposadr[joint]))
            if name != 'phone':
                self.support_body_ids[name] = body_id
        self.gripper_eef_body_id = self.sim.model.body_name2id("gripper0_right_eef")
        self.phone_grasp_assist_id = mujoco.mj_name2id(
            self.sim.model._model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            "phone_grasp_assist",
        )

    def set_phone_grasp_assist(self, active):
        """Latch the current physical grasp pose, or release it before scoring."""
        active = bool(active)
        if active:
            data, model = self.sim.data, self.sim.model
            first = self.gripper_eef_body_id
            second = self.phone_body_id
            rotation_first = data.body_xmat[first].reshape(3, 3)
            rotation_second = data.body_xmat[second].reshape(3, 3)
            relative_position = rotation_first.T @ (
                data.body_xpos[second] - data.body_xpos[first]
            )
            relative_rotation = rotation_first.T @ rotation_second
            relative_quaternion = np.empty(4, dtype=float)
            mujoco.mju_mat2Quat(relative_quaternion, relative_rotation.reshape(-1))
            model.eq_data[self.phone_grasp_assist_id, 3:6] = relative_position
            model.eq_data[self.phone_grasp_assist_id, 6:10] = relative_quaternion
        self.sim.data.eq_active[self.phone_grasp_assist_id] = active
        self.sim.forward()

    def _reset_internal(self):
        super()._reset_internal()
        # Override the generic PiPER-X home pose for this task before placing
        # objects.  This is a task-local pose change; it does not alter the
        # shared robot asset or any other task's initialization.
        for robot in self.robots:
            indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
            if len(indexes) % len(SAFE_ROBOT_ARM_QPOS) != 0:
                raise ValueError('unexpected PiPER-X arm joint layout')
            self.sim.data.qpos[indexes] = np.tile(
                SAFE_ROBOT_ARM_QPOS, len(indexes) // len(SAFE_ROBOT_ARM_QPOS))
        # Reset replays the SAME explicit Task + Episode instance. Request a
        # new episode_seed to create a new layout; there is no hidden RNG state.
        self.sim.data.eq_active[self.phone_grasp_assist_id] = False
        self.layout_positions = {}
        for name, xyz in self.episode_instance["position_xyz_m"].items():
            qpos = getattr(self, name + "_qpos")
            self.sim.data.qpos[qpos:qpos + 3] = xyz
            self.sim.data.qpos[qpos + 3:qpos + 7] = self.episode_instance["quaternion_wxyz"][name]
            self.layout_positions[name] = np.asarray(xyz[:2], dtype=float)
        self.sim.data.qvel[:] = 0.0
        self.sim.forward()
        check_initial_contacts(self.sim.model._model, self.sim.data._data, names=("phone", *self.support_targets))
        # Agent time starts after 25 physical settling substeps.
        for _ in range(25):
            self.sim.step()
        self.sim.data.time = 0.0
        self.sim.forward()
        self._phone_initial_z = float(self.sim.data.body_xpos[self.phone_body_id, 2])
        self.sim.data.eq_active[self.phone_grasp_assist_id] = False
        self._phone_lift_reference_z = self._phone_initial_z
        self._phone_lifted = False
        self._successful_supports: set[str] = set()
        self._score = 0.0
        self._lean_valid_windows = {name: [] for name in self.support_targets}
        self._lean_position_windows = {name: [] for name in self.support_targets}
        self._lean_cup_tilt_windows = {name: [] for name in self.support_targets}
        self._lean_valid_fractions = {name: 0.0 for name in self.support_targets}
        self._lean_drifts = {name: 0.0 for name in self.support_targets}
        self._last_lean_samples = {
            name: self._empty_lean_sample() for name in self.support_targets
        }

    def _body_tilt_deg(self, body_id):
        rotation = self.sim.data.body_xmat[body_id].reshape(3, 3)
        return math.degrees(math.acos(np.clip(rotation[2, 2], -1.0, 1.0)))

    @staticmethod
    def _empty_lean_sample():
        return {
            "screen_up": False,
            "upright_lean": False,
            "back_support": False,
            "front_support": False,
            "home_button_down": False,
            "top_up": False,
            "bottom_table": False,
            "gripper_contact": False,
            "linear_speed": float("inf"),
            "angular_speed": float("inf"),
        }

    def lean_sample(self, target=None):
        """Measure every physical condition used to accept a released lean.

        Local ``-X`` is the screen, ``+X`` is the back, and ``+Z`` points to
        the phone's top edge. Contacts count only when MuJoCo reports a
        load-bearing normal force, so proximity alone cannot pass the task.
        """
        target = self.support_target if target is None else str(target)
        if target not in self.support_targets:
            raise ValueError(f"unknown support target {target!r}")
        model = self.sim.model._model
        data = self.sim.data._data
        return measure_lean(model, data, target)

    @staticmethod
    def _sample_is_valid(sample):
        return sample_is_valid(sample)

    def _update_lean_evaluation(self, target, sample, phone_position):
        """Require ten uninterrupted seconds; any invalid tick restarts the timer."""
        window_size = int(round(LEAN_STABLE_SECONDS * self.control_freq)) + 1
        valid_window = self._lean_valid_windows[target]
        position_window = self._lean_position_windows[target]
        tilt_window = self._lean_cup_tilt_windows[target]
        cup_tilt = self._body_tilt_deg(self.support_body_ids[target])
        valid = bool(
            self._sample_is_valid(sample)
            and not bool(self.sim.data.eq_active[self.phone_grasp_assist_id])
        )
        if not valid:
            valid_window.clear(); position_window.clear(); tilt_window.clear()
            self._lean_valid_fractions[target] = 0.0
            self._lean_drifts[target] = 0.0
            return 0.0
        valid_window.append(True)
        position_window.append(np.asarray(phone_position, dtype=float).copy())
        tilt_window.append(cup_tilt)
        if len(valid_window) > window_size:
            valid_window.pop(0)
            position_window.pop(0)
            tilt_window.pop(0)
        positions = np.asarray(position_window)
        drift = float(np.max(np.linalg.norm(positions - positions[-1], axis=1)))
        self._lean_drifts[target] = drift
        if drift > LEAN_MAX_DRIFT_M:
            valid_window.clear(); position_window.clear(); tilt_window.clear()
            self._lean_valid_fractions[target] = 0.0
            return 0.0
        self._lean_valid_fractions[target] = 1.0
        success = len(valid_window) == window_size
        if success and not self._successful_supports:
            self._successful_supports.add(target)
            self._score = SCORE_MAX
            self.support_target = target
            self._phone_lifted = False
            self._phone_lift_reference_z = float(phone_position[2])
            return POINTS_PER_SUPPORT
        return 0.0

    def _phone_metrics(self):
        pos = np.asarray(self.sim.data.body_xpos[self.phone_body_id])
        cup = np.asarray(self.sim.data.body_xpos[self.support_body_ids[self.support_target]])
        return float(np.linalg.norm(pos[:2] - cup[:2])), float(pos[2])

    def _task_instruction(self):
        return (
            "Pick up the phone lying screen-up directly on the table using either gripper, "
            "then leave it stably leaning against any one of the four selected objects. "
            "You may push, drag, grasp, and reposition the phone or supports to clear access. "
            "There are exactly two large and two small objects. All four objects are eligible to score, "
            "but their curved, light, or narrow geometry may make a stable lean impossible. "
            "The front is the black screen, the back is light blue, and the circular Home button is on the lower half of the front. "
            "Keep the Home-button end down: the bottom-to-top axis must be 45-75 degrees above the tabletop. "
            "The light-blue back must have load-bearing support contact, the black screen must face away "
            "from that support without load-bearing front contact, and the rounded bottom edge must "
            "contact the table after gripper release. Hold a valid stable lean for at least "
            "ten consecutive seconds; any invalid sample or excessive drift restarts the timer. "
            "A valid lean against any selected object earns 100/100 and ends "
            "the episode; otherwise the score is zero. Support kinds, role identities, sizes, masses, and friction stay fixed within a Task Instance; "
            "only initial positions vary between Episodes. "
            "All sampled properties stay fixed within an Episode. Do not reset during "
            "an attempt; objects subsequently move only through physical interaction."
        )

    def get_episode_language_prompt(self):
        from react_manip.core.language import task_language_prompt
        return task_language_prompt(
            self, task_id='phone_lean',
            task_instruction=self._task_instruction() +
                f" Time limit: {self.horizon / self.control_freq:g} seconds. "
                "Linear speed must be <=0.005 m/s, angular speed <=0.05 rad/s, and drift <=0.005 m. "
                "No fixed camera-facing or robot-facing direction is required.",
            success_condition="Any selected object has load-bearing contact with the light-blue phone back; the black screen has no load-bearing contact with that support; rounded bottom contacts table; grasp assist off and no load-bearing gripper contact; phone long-axis angle is 45-75 degrees inclusive; all speed and drift limits hold for ten continuous seconds. Success earns 100 and ends the episode; otherwise 0.",
        )

    def instance_manifest(self):
        return copy.deepcopy({
            "task_instance": self.task_instance,
            "episode_instance": self.episode_instance,
        })

    def reward(self, action=None):
        del action
        return 0.0

    def _check_success(self):
        return bool(self._successful_supports)

    def _post_action(self, action):
        _, done, info = super()._post_action(action)
        info = dict(info or {})
        distance, height = self._phone_metrics()
        finite = bool(
            np.isfinite(self.sim.data.qpos).all() and np.isfinite(self.sim.data.qvel).all()
        )
        self._phone_lifted |= bool(height > self._phone_lift_reference_z + 0.04)
        samples = {
            target: self.lean_sample(target) if finite else self._empty_lean_sample()
            for target in self.support_targets
        }
        self._last_lean_samples = samples
        score_delta = 0.0
        awarded_target = None
        for target in self.support_targets:
            delta = self._update_lean_evaluation(
                target, samples[target], self.sim.data.body_xpos[self.phone_body_id]
            )
            if delta:
                score_delta += delta
                awarded_target = target
                break
        success = self._check_success()
        events = ([f"phone_stably_leaned:{awarded_target}"] if awarded_target else [])
        failure_reason = None
        if not finite:
            failure_reason = "non_finite_state"
        elif done and not success:
            failure_reason = "time_limit"
        criteria_by_support = {}
        for target in self.support_targets:
            criteria_by_support[target] = {
                **samples[target],
                "already_scored": target in self._successful_supports,
                "qualified_fraction": self._lean_valid_fractions[target],
                "required_fraction": LEAN_REQUIRED_FRACTION,
                "settle_drift_m": self._lean_drifts[target],
                "max_settle_drift_m": LEAN_MAX_DRIFT_M,
                "evaluation_window_sec": min(
                    LEAN_STABLE_SECONDS,
                    max(0, len(self._lean_valid_windows[target])-1) / float(self.control_freq),
                ),
            }
        report_target = awarded_target or self.support_target
        criteria = criteria_by_support[report_target]
        successful_supports = [
            target for target in self.support_targets if target in self._successful_supports
        ]
        info.update({
            "phone_lean_xy_error": distance,
            "phone_height": height,
            "support_target": report_target,
            "support_names": list(self.support_targets),
            "awarded_target": awarded_target,
            "successful_supports": successful_supports,
            "remaining_supports": [
                target for target in self.support_targets if target not in self._successful_supports
            ],
            "score": self._score,
            "total_score": self._score,
            "score_delta": score_delta,
            "support_score": self._score,
            "support_score_max": SCORE_MAX,
            "trial_score_max": POINTS_PER_SUPPORT,
            "criteria": criteria,
            "criteria_by_support": criteria_by_support,
            "layout_seed": self.layout_seed,
            "task_seed": self.task_seed,
            "episode_seed": self.episode_seed,
            "task_instance_id": self.task_instance["task_instance_id"],
            "episode_instance_id": self.episode_instance["episode_instance_id"],
            "layout_positions": {
                key: value.tolist() for key, value in self.layout_positions.items()
            },
            "events": events,
            "failure": (
                {"recoverable": False, "reason": failure_reason}
                if failure_reason else None
            ),
            "success_state": success,
            "terminal_success": success,
        })
        return float(score_delta), bool(done or success or failure_reason), info


def make_env(seed=0, robot_id="dual_piperx", **kwargs):
    """Build exactly the requested seeds; never replace an unsolvable scene."""
    kwargs.setdefault('task_seed', seed)
    kwargs.setdefault('episode_seed', 0)
    return PhoneLean(robots=robot_id, seed=seed, **kwargs)
