#!/usr/bin/env python3
"""Restore pinned historical follower models with matching visual/contact meshes.

Generated models use upstream inertias / limits, but servo gains are simulation
defaults, NOT system-identified hardware parameters. No ROS dependency.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from react_manip.robots.agilex_description import extract_followers

REV = "f799ae0192c0c2bbd502ec4d9bbec8c2633acd1a"
BASE = f"https://raw.githubusercontent.com/agilexrobotics/mobile_aloha_sim/{REV}/"


def quat(rpy):
    r, p, y = np.fromstring(rpy, sep=" ") / 2
    cr, cp, cy, sr, sp, sy = np.cos(r), np.cos(p), np.cos(y), np.sin(r), np.sin(p), np.sin(y)
    return " ".join(map(str, [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                              cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]))


def pose(origin):
    return {"pos": origin.get("xyz", "0 0 0"), "quat": quat(origin.get("rpy", "0 0 0"))} if origin is not None else {}


def build(destination, offline=False):
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    lock_path = Path(__file__).with_name("legacy_agilex_assets.lock.json")
    lock = json.loads(lock_path.read_text())

    manifest_path = destination / 'provenance.json'
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text())
        if (recorded.get('source_lock_sha256') == hashlib.sha256(lock_path.read_bytes()).hexdigest()
                and all((destination/n).is_file() and hashlib.sha256((destination/n).read_bytes()).hexdigest() == h
                        for n,h in recorded.get('generated_sha256', {}).items())
                and recorded.get('generated_sha256')
                and all((destination/'source'/n).is_file() and hashlib.sha256((destination/'source'/n).read_bytes()).hexdigest() == h
                        for n,h in lock['files'].items())):
            print('Verified cached legacy assets:', destination, flush=True)
            return

    def fetch(path):
        target = destination / "source" / path
        if path not in lock["files"]:
            raise ValueError(f"Unpinned legacy source: {path}")
        if not target.exists():
            if offline:
                raise FileNotFoundError(f"Offline source missing: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            print("download", path, flush=True)
            with urllib.request.urlopen(BASE + path, timeout=60) as response:
                data = response.read()
            target.write_bytes(data)
        hashes[path] = hashlib.sha256(target.read_bytes()).hexdigest()
        if hashes[path] != lock["files"][path]:
            raise ValueError(f"Legacy source checksum mismatch: {path}")
        return target

    src = ET.parse(fetch("aloha_new_description/urdf/aloha_new.urdf")).getroot()
    fetch("aloha_new_description/package.xml")
    # Structural validation also ensures leaders cannot enter the model.
    extract_followers(src, "agilex_aloha")
    links = {l.get("name"): l for l in src.findall("link")}
    joints = {j.get("name"): j for j in src.findall("joint")}

    def new_model(name):
        root = ET.Element("mujoco", model=name)
        ET.SubElement(root, "compiler", angle="radian")
        ET.SubElement(root, "asset")
        ET.SubElement(root, "worldbody")
        ET.SubElement(root, "actuator")
        return root

    converted = {}

    def add_link(root, parent, source_name, name, transform):
        link = links[source_name]
        body = ET.SubElement(parent, "body", name=name, **transform)
        inertial = link.find("inertial")
        inertia = inertial.find("inertia")
        origin = inertial.find("origin")
        # Current source uses identity inertia frames. Fail on changes instead
        # of silently interpreting a rotated tensor in the wrong frame.
        if origin is not None and not np.allclose(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "), 0):
            raise ValueError("rotated inertial frame requires tensor conversion")
        ET.SubElement(body, "inertial", pos=origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
                      mass=inertial.find("mass").get("value"),
                      fullinertia=" ".join(inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
        for index, collision in enumerate(link.findall("collision")):
            mesh = collision.find("geometry/mesh")
            path = mesh.get("filename").removeprefix("package://")
            if not path.lower().endswith(".stl"):
                raise ValueError(f"expected official STL collision mesh, got {path}")
            mesh_path = fetch(path)
            if path not in converted:
                import trimesh
                mesh_data = trimesh.load(str(mesh_path), force='mesh')
                if name.startswith('finger'):
                    import coacd
                    parts = coacd.run_coacd(coacd.Mesh(mesh_data.vertices, mesh_data.faces),
                        threshold=.01, max_convex_hull=128, preprocess_resolution=150,
                        resolution=2000, mcts_iterations=20, mcts_nodes=10, mcts_max_depth=3, seed=0)
                    pieces = [trimesh.Trimesh(v, f).convex_hull for v, f in parts]
                else:
                    pieces = [mesh_data.convex_hull]
                converted[path] = []
                for part_index, piece in enumerate(pieces):
                    target = destination / (mesh_path.stem + f'_convex_{part_index}.obj')
                    piece.export(target)
                    converted[path].append(target.name)
            for part_index, filename in enumerate(converted[path]):
                suffix = f'{index}_{part_index}'
                mesh_name = name + '_mesh' + suffix
                ET.SubElement(root.find('asset'), 'mesh', name=mesh_name, file=filename,
                              scale=mesh.get('scale', '1 1 1'))
                attrs = dict(type='mesh', mesh=mesh_name, **pose(collision.find('origin')))
                ET.SubElement(body, 'geom', name=name+'_collision'+suffix, group='0',
                              friction='0.8 0.005 0.0001', **attrs)
                ET.SubElement(body, 'geom', name=name+'_visual'+suffix, group='1',
                              contype='0', conaffinity='0', rgba='0.75 0.78 0.82 1', **attrs)
        return body

    for robot, arms in (("legacy_piperx", [("right", "fr")]), ("aloha", [("right", "fr"), ("left", "fl")]),
                        ("aloha_mobile", [("right", "fr"), ("left", "fl")])):
        root = new_model(robot)
        contact = ET.SubElement(root, "contact")
        base = ET.SubElement(root.find("worldbody"), "body", name="base")
        if robot == "aloha_mobile":
            from scripts.agilex_wheeled_builder import add_wheeled_base
            mobile_metadata = add_wheeled_base(root, base, src, fetch, destination, pose)
        for arm, prefix in arms:
            mount = joints[prefix + "_base_joint"].find("origin")
            mount_xyz = np.fromstring(mount.get("xyz", "0 0 0"), sep=" ")
            if robot == "legacy_piperx":
                mount_xyz[:2] = 0
            arm_parent = base.find("body[@name='mobile_body_Link']") if robot == "aloha_mobile" else base
            transform = pose(mount) if robot == "aloha_mobile" else {"pos": f"{mount_xyz[0]} {mount_xyz[1]} 0"}
            parent = add_link(root, arm_parent, prefix+"_base_link", arm+"_base", transform)
            ET.SubElement(parent, "site", name=arm+"_center", size="0.002", rgba="0 0 0 0")
            for i in range(1, 7):
                joint = joints[f"{prefix}_joint{i}"]
                name = f"{arm}_joint{i}"
                parent = add_link(root, parent, f"{prefix}_link{i}",
                                  f"{arm}_link{i}" if i < 6 else f"{arm}_hand", pose(joint.find("origin")))
                limit = joint.find("limit")
                ET.SubElement(parent, "joint", name=name, type="hinge", axis=joint.find("axis").get("xyz"),
                              limited="true", range=limit.get("lower")+" "+limit.get("upper"), damping="0.1")
                effort = float(limit.get("effort"))
                ET.SubElement(contact, "exclude", body1=arm+"_base" if i == 1 else f"{arm}_link{i-1}",
                              body2=f"{arm}_link{i}" if i < 6 else f"{arm}_hand")
                ET.SubElement(root.find("actuator"), "motor", name=name+"_motor", joint=name,
                              ctrllimited="true", ctrlrange=f"{-effort} {effort}")
        # Replace the operator display at the top of ALOHA / Mobile ALOHA
        # with a forward-facing head camera. The camera is attached to the
        # embodiment (not the task arena), so it follows a mobile base and is
        # exposed through the same compiled-model API as wrist views.
        if robot in ("aloha", "aloha_mobile"):
            camera_parent = base.find("body[@name='mobile_body_Link']") if robot == "aloha_mobile" else base
            # Define the sensor basis explicitly rather than relying on a
            # quaternion.  MuJoCo cameras look along local -Z; these axes make
            # local -Z point along the robot's +X (forward) and local +Y be
            # vertical +Z (image-up).  The previous single-axis orientation
            # left image-up along the body Y axis, producing a 90-degree roll.
            # ``xyaxes`` is also easier to audit in the compiled model than a
            # quaternion and keeps the same orientation for fixed and mobile
            # embodiments (the mobile base yaw is applied by its parent body).
            ET.SubElement(camera_parent, "camera", name="head_camera", pos="0.24 0 1.24",
                          xyaxes="0 -1 0  0 0 1", fovy="70")
        ET.indent(root)
        ET.ElementTree(root).write(destination / f"{robot}.xml", encoding="utf-8")

    root = new_model("PiperGripper")
    base = ET.SubElement(root.find("worldbody"), "body", name="gripper_base")
    for i in (7, 8):
        joint = joints[f"fr_joint{i}"]
        body = add_link(root, base, f"fr_link{i}", f"finger{i}", pose(joint.find("origin")))
        limit = joint.find("limit")
        ET.SubElement(body, "joint", name=f"finger_joint{i}", type="slide", axis=joint.find("axis").get("xyz"),
                      limited="true", range=limit.get("lower")+" "+limit.get("upper"), damping="8")
        effort = float(limit.get("effort"))
        ET.SubElement(root.find("actuator"), "position", name=f"finger_actuator{i}", joint=f"finger_joint{i}",
                      kp="300", ctrllimited="true", ctrlrange=limit.get("lower")+" "+limit.get("upper"),
                      forcelimited="true", forcerange=f"{-effort} {effort}")
    # The two fingers are opposing slides. Excluding their self-collision lets
    # the position actuators reach the closed target while retaining contacts
    # between each finger and manipulated objects.
    contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", body1="finger7", body2="finger8")
    # Tool center is a documented task-control convention in link6 frame.
    eef = ET.SubElement(base, "body", name="eef", pos="0 0 0.135")
    for name in ("grip_site", "grip_site_cylinder", "ee", "ee_x", "ee_y", "ee_z"):
        ET.SubElement(eef, "site", name=name, size="0.002", rgba="0 0 0 0", group="1")
    # Upper/rear wrist mount, looking 6 cm beyond TCP in this local frame.
    # Explicit image-right / image-up axes avoid hidden roll/mirror corrections.
    ET.SubElement(eef, "camera", name="eye_in_hand", pos="0 0.06 -0.08",
                  xyaxes="-1 0 0  0 .919145030018 .393919298579", fovy="65")
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "force", name="force_ee", site="ee")
    ET.SubElement(sensors, "torque", name="torque_ee", site="ee")
    ET.indent(root)
    ET.ElementTree(root).write(destination / "gripper.xml", encoding="utf-8")
    # Make imported base mesh references relocatable too.
    used = set()
    for filename in ('legacy_piperx.xml', 'aloha.xml', 'aloha_mobile.xml', 'gripper.xml'):
        tree = ET.parse(destination / filename)
        for mesh in tree.findall('asset/mesh'):
            value = Path(mesh.get('file'))
            if value.is_absolute():
                mesh.set('file', str(value.relative_to(destination.resolve())))
            used.add(mesh.get('file'))
        ET.indent(tree)
        tree.write(destination / filename, encoding='utf-8')
        used.add(filename)
    gripper = ET.parse(destination / 'gripper.xml')
    names = {f'finger{i}': [g.get('name') for g in gripper.iter('geom')
                          if g.get('name', '').startswith(f'finger{i}_collision')] for i in (7, 8)}
    (destination / 'provenance.json').write_text(json.dumps(dict(
        schema='legacy-agilex-family.v1', source_revision=REV,
        source_lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        generated_sha256={n: hashlib.sha256((destination/n).read_bytes()).hexdigest() for n in sorted(used)},
        gripper_collision_names=names,
        collision_policy='Identical convex meshes for rendering and contact; no hidden finger pads',
        warning='Historical follower structure; collision repair means old snapshots require regeneration'
    ), indent=2)+'\n')
    print("Built", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/robots/agilex/legacy")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    build(args.output, args.offline)
