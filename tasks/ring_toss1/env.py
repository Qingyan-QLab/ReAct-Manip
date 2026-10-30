"""Free rigid rings, physical release, and stable encirclement scoring."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from react_manip.visual_task import ManipulationTask
from react_manip.robots.agilex_robosuite import controller_config, resolve_robot
from tasks.ring_toss1.instances import PROFILE_PATH, resolve_instance, sample_task_instance
import json

ROOT = Path(__file__).resolve().parent
SOURCE_RING_RADIUS = .0875


class TossArena(Arena):
    def __init__(self, instance):
        super().__init__(str(ROOT / "assets/source_scene.xml"))
        ring_radius = instance["ring_centerline_radius_m"]
        tube_radius = instance["ring_tube_radius_m"]
        table_z = instance["table_top_z_m"]
        self.root.find("option").set("integrator", "implicitfast")
        self.root.find("option").set("iterations", "100")
        # Keep the template's hollow capsule ring and build the sampled scene.
        for geom in list(self.worldbody.findall("geom")):
            self.worldbody.remove(geom)
        for body in list(self.worldbody.findall("body")):
            if not body.get("name", "").startswith("ring_"):
                continue
            index = int(body.get("name").split("_")[-1])
            if index != 0:
                self.worldbody.remove(body); continue
            x, y = instance["pickup_xy_m"]
            body.set("pos", f"{x} {y} {table_z + .05 + tube_radius}")
            ET.SubElement(body, "freejoint", name=f"ring_{index}_free")
            mass = instance["ring_mass_kg"]
            planar = mass * (ring_radius**2 / 2 + 5*tube_radius**2/8)
            polar = mass * (ring_radius**2 + 3*tube_radius**2/4)
            ET.SubElement(body, "inertial", pos="0 0 0", mass=str(mass),
                          diaginertia=f"{planar} {planar} {polar}")
            for geom in body.findall("geom"):
                # Keep the visible tube and contact surface identical.
                endpoints = [float(v) * ring_radius / SOURCE_RING_RADIUS for v in geom.get("fromto").split()]
                geom.set("fromto", " ".join(map(str, endpoints)))
                geom.set("size", str(tube_radius))
                geom.set("group", "1")
                geom.set("friction", " ".join(map(str, instance["ring_friction"])))
                geom.set("condim", "4")
                geom.set("solref", ".01 1")
                geom.set("margin", "0")
            # Two visible supports raise the ring enough for fingers to reach
            # underneath, without attaching or pinning it to the supply area.
            for sign in (-1, 1):
                ET.SubElement(self.worldbody, "geom", name=f"rest_{index}_{sign}", type="cylinder",
                              pos=f"{x+sign*ring_radius} {y} {table_z+.025}", size=".013 .025",
                              group="1", rgba=".27 .30 .32 1")
        textures = ROOT.parent / "laptop_into_drawer/assets/textures"
        for name, filename in (("wood", "light-wood.png"), ("floor", "light-gray-floor-tile.png"),
                               ("wall", "white-plaster.png")):
            ET.SubElement(self.asset, "texture", name="toss_"+name, type="2d", file=str(textures/filename))
            ET.SubElement(self.asset, "material", name="toss_"+name, texture="toss_"+name, texrepeat="4 4")
        self.geom("table", f"0 .30 {table_z-.025}", ".72 1.05 .025", material="toss_wood")
        self.geom("floor", "0 0 -.05", "3 3 .05", material="toss_floor")
        for i,(x,y) in enumerate(((-.62,-.65),(.62,-.65),(-.62,1.25),(.62,1.25))):
            self.geom(f"leg_{i}", f"{x} {y} .315", ".035 .035 .315", rgba=".2 .23 .27 1")
        for name,pos,size in (("back","0 3 1.5","3 .05 1.5"),("front","0 -3 1.5","3 .05 1.5"),
                              ("left","-3 0 1.5",".05 3 1.5"),("right","3 0 1.5",".05 3 1.5")):
            self.geom(name,pos,size,material="toss_wall")
        colors = (".18 .55 .65 1", ".28 .4 .80 1", ".85 .58 .12 1")
        for i,(x,y,r,h) in enumerate(instance["targets"]):
            ET.SubElement(self.worldbody,"geom",name=f"target_{i}",type="cylinder",pos=f"{x} {y} {table_z+h/2}",
                          size=f"{r} {h/2}",group="1",rgba=colors[i//3],friction=".6 .005 .0001")
        for i,(name,pos,axes) in enumerate((
            ("toss_overview","1.8 -2.1 2.25",".8 .6 0 -.3 .4 .866"),
            ("toss_top","0 .3 3.2","1 0 0 0 1 0"),
            ("toss_pickup",".65 -1.1 1.25",".75 .66 0 -.3 .34 .89"))):
            ET.SubElement(self.worldbody,"camera",name=name,pos=pos,xyaxes=axes,fovy="55")
        ET.SubElement(self.worldbody,"light",name="fill",pos="-1 1 2.5",dir=".2 -.2 -1",
                      diffuse=".4 .4 .4",ambient=".12 .12 .12",castshadow="false")

        from react_manip.cameras import ensure_scene_cameras
        ensure_scene_cameras(self, 'ring_toss1')

    def geom(self,name,pos,size,**kwargs):
        ET.SubElement(self.worldbody,"geom",name=name,type="box",pos=pos,size=size,group="1",**kwargs)


class RingToss1(ManipulationEnv):
    def __init__(self, seed=0, robot_id='dual_piperx', scenario=None, **kwargs):
        from react_manip.core.scenarios import Scenario
        from react_manip.instances.records import validate_parameters
        if robot_id not in ('dual_piperx', 'piperx', 'aloha', 'legacy_piperx', 'panda_libero'):
            raise ValueError('ring_toss1 supports dual_piperx and piperx')
        self.robot_id = robot_id
        profile = json.loads(PROFILE_PATH.read_text())
        if scenario is None:
            scenario = Scenario('2.0', 'ring_toss1', f'ring_toss1_{seed}', seed,
                                sample_task_instance(seed, profile), profile['assets'])
        if scenario.task_id != 'ring_toss1':
            raise ValueError('ring scenario task mismatch')
        validate_parameters(scenario.parameters, {'task_id': 'ring_toss1', 'sampling_profile': profile})
        self.scenario = scenario
        self.instance = resolve_instance(scenario.parameters, seed=scenario.seed, profile=profile)
        self.targets = self.instance['targets']
        self.points = self.instance['points']
        self.num_rings = 1
        config = controller_config(('right', 'left') if robot_id in ('dual_piperx', 'aloha') else ('right',))
        for arm in config['body_parts'].values():
            arm['output_max'] = [.04]*3 + [.10]*3
            arm['output_min'] = [-.04]*3 + [-.10]*3
            arm['kp'] = 150
        if kwargs.pop('initialization_noise', None) is not None:
            raise ValueError('ring initialization noise is disabled')
        if robot_id=='panda_libero':
            from react_manip.robots.resolution import resolve_robosuite_robot
            resolved=resolve_robosuite_robot(robot_id)
            config=resolved.controller_configs
            kwargs['gripper_types']=resolved.gripper_types
            model_name=resolved.robot_name
        else:model_name=resolve_robot(robot_id)
        super().__init__(robots=model_name,controller_configs=config,
                         has_renderer=kwargs.pop('has_renderer',False),
                         has_offscreen_renderer=kwargs.pop('has_offscreen_renderer',False),
                         use_camera_obs=kwargs.pop('use_camera_obs',False),
                         camera_names='toss_overview',render_camera='toss_overview',
                         initialization_noise=None,control_freq=20,horizon=3600,seed=scenario.seed,**kwargs)

    def _task_instruction(self):
        return (
            "There is one ring and nine target pegs. "
            "Grasp one ring, lift it clear of the support, and throw it toward "
            "a target peg. "
            "A ring scores only when it passes around the peg, leaves the gripper, "
            "and settles stably. "
            f"Lift the ring center above {self.instance['task']['lift_height_m']:g} m while grasped, then accumulate "
            f"at least {self.instance['task']['flight_sec']:g} seconds free of gripper and support contact. "
            f"Settle while supported and released for {self.instance['task']['settle_sec']:g} seconds, "
            f"with linear speed <{self.instance['task']['stable_linear_speed_m_s']:g} m/s and angular speed "
            f"<{self.instance['task']['stable_angular_speed_rad_s']:g} rad/s. "
            "The near, middle, and far rows score 60, 80, and 100 points. "
            "Any score is success; each ring scores once, with no revocation or upgrade. "
            f"The episode continues until {self.horizon / self.control_freq:g} seconds."
        )

    def get_episode_language_prompt(self):
        from react_manip.core.language import task_language_prompt
        return task_language_prompt(
            self, task_id='ring_toss1',
            task_instruction=self._task_instruction(),
            success_condition='a physically lifted and released ring stably encircles any target peg',
            constraints=(),
        )

    def _load_model(self):
        super()._load_model()
        from react_manip.robots.mounts import use_existing_mount, set_fixed_mount_pose
        robot=self.robots[0].robot_model
        if self.robot_id in ('dual_piperx', 'aloha'):
            set_fixed_mount_pose(robot, position=[-.30,-.68-.23875,.68], yaw=math.pi/2, height=.68)
        else:
            use_existing_mount(robot)
            robot.set_base_xpos([0,-.68,0. if self.robot_id=='panda_libero' else .68]);robot.set_base_ori(np.array([0,0,math.pi/2]))
        # Contact geometry is supplied by the central PiPER-X asset.
        self.model=ManipulationTask(mujoco_arena=TossArena(self.instance),mujoco_robots=[robot],mujoco_objects=[])

    def _setup_references(self):
        super()._setup_references()
        m=self.sim.model
        self.ring_ids=[m.body_name2id("ring_0")]
        self.ring_qpos=[int(m.jnt_qposadr[m.body_jntadr[b]]) for b in self.ring_ids]

    def _reset_internal(self):
        super()._reset_internal()
        self.lifted=[False];self.flight=[0.];self.settled=[0.]
        self.awarded={};self.total_score=0;self.last_target=None

    def encircled_target(self,index):
        """Peg axis must pass through the ring's hole, below the peg tip."""
        d=self.sim.data;b=self.ring_ids[index]
        p=d.body_xpos[b];normal=d.body_xmat[b].reshape(3,3)[:,2]
        if abs(normal[2])<.8:return None
        table_z=self.instance["table_top_z_m"]
        tube_radius=self.instance["ring_tube_radius_m"]
        inner_radius=self.instance["inner_radius_m"]
        for j,(x,y,r,h) in enumerate(self.targets):
            z=p[2]-(normal[0]*(x-p[0])+normal[1]*(y-p[1]))/normal[2]
            intersection=np.array([x,y,z])
            if (table_z<z<table_z+h-tube_radius
                    and np.linalg.norm(intersection-p)+r/abs(normal[2])<inner_radius):
                return j
        return None

    def reward(self,action=None):
        return 0.

    def _post_action(self,action):
        _,done,info=super()._post_action(action)
        m,d=self.sim.model,self.sim.data
        gripper_contact=set();supported=set()
        for c in d.contact[:d.ncon]:
            bodies=[int(m.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
            for b,other in (bodies,bodies[::-1]):
                if b in self.ring_ids:
                    index=self.ring_ids.index(b)
                    if m.body_names[other].startswith('gripper0_'):gripper_contact.add(index)
                    else:supported.add(index)
        delta=0;events=[]
        task=self.instance["task"]
        for i,b in enumerate(self.ring_ids):
            if i in self.awarded:continue
            if i in gripper_contact and d.body_xpos[b,2]>task["lift_height_m"]:self.lifted[i]=True
            if self.lifted[i] and i not in gripper_contact and i not in supported:
                self.flight[i]+=self.control_timestep
            target=self.encircled_target(i)
            stable=(np.linalg.norm(d.get_body_xvelp(f'ring_{i}'))<task["stable_linear_speed_m_s"]
                    and np.linalg.norm(d.get_body_xvelr(f'ring_{i}'))<task["stable_angular_speed_rad_s"])
            valid=(self.lifted[i] and self.flight[i]>=task["flight_sec"]
                   and i not in gripper_contact and i in supported and stable
                   and target is not None)
            self.settled[i]=self.settled[i]+self.control_timestep if valid else 0.
            if self.settled[i]+1e-12>=task["settle_sec"]:
                self.awarded[i]=target;delta+=self.points[target];self.last_target=target
                events.append(f'ring_{i}_target_{target}')
        self.total_score+=delta
        info.update(score=self.total_score,total_score=self.total_score,score_delta=delta,
                    hits=len(self.awarded),last_target=self.last_target,events=events,
                    ring_states=["scored" if i in self.awarded else "airborne/landed"
                                 if self.flight[i]>=task["flight_sec"] else "lifted"
                                 if self.lifted[i] else "ready" for i in range(self.num_rings)],
                    task_seed=self.instance["seed"],
                    success_state=bool(self.awarded),terminal_success=bool(self.awarded))
        info["termination_reason"] = "timeout" if done else None
        return float(delta),done,info
