"""Official aloha_new URDF chassis tree -> free-root MJCF.

CAD and inertias come from pinned upstream. The operator display bundled in the
mobile body CAD is intentionally omitted; a head camera is added by the outer
builder at the same upper-body station. Wheel servo/friction are explicitly
simulation tuning, not system identification. No planar propulsion actuators.
"""
import xml.etree.ElementTree as ET
import numpy as np


def add_wheeled_base(root, base, source, fetch, destination, pose):
    import trimesh
    links = {x.get("name"): x for x in source.findall("link")}
    joints = source.findall("joint")
    ET.SubElement(base, "freejoint", name="mobile_free")
    bounds = {}

    def add(link_name, parent, origin):
        link = links[link_name]
        name = "mobile_" + link_name
        body = ET.SubElement(parent, "body", name=name, **pose(origin))
        if link_name in ("left_wheel_link", "right_wheel_link"):
            xyz = np.fromstring(body.get("pos"), sep=" ")
            xyz[2] -= 0.011
            body.set("pos", " ".join(map(str, xyz)))
        inertial = link.find("inertial")
        # Upstream puts chassis inertia in an identity fixed child. Merge it
        # onto the chassis body so MuJoCo never infers mass from mass-zero CAD.
        if link_name == "base_link":
            inertial = links["inertial_link"].find("inertial")
        if inertial is not None:
            inertia_origin = inertial.find("origin")
            if inertia_origin is not None and not np.allclose(np.fromstring(inertia_origin.get("rpy", "0 0 0"), sep=" "), 0):
                raise ValueError("rotated chassis inertia requires tensor conversion")
            inertia = inertial.find("inertia")
            ET.SubElement(body, "inertial", mass=inertial.find("mass").get("value"),
                          pos=inertial.find("origin").get("xyz", "0 0 0"),
                          fullinertia=" ".join(inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
        for i, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            path = mesh.get("filename").removeprefix("package://")
            raw = fetch(path)
            converted = destination / (name + f"_{i}.obj")
            scene = trimesh.load(str(raw), force="scene")
            geometry = scene.to_geometry()
            # The upstream mobile CAD stores the operator display and frame
            # in one ``body_Link`` mesh. The benchmark replaces that display
            # with a head camera, so clip only the elevated panel (z~=1.18--
            # 1.34 m in this link frame) while preserving the chassis below.
            # Applying the filter before exporting and collision decomposition
            # keeps visual and physical geometry consistent.
            if link_name == "body_Link":
                geometry.update_faces(geometry.triangles_center[:, 2] < 1.15)
                geometry.remove_unreferenced_vertices()
            geometry.export(str(converted))
            bounds[link_name] = geometry.bounds.tolist()
            mesh_name = name + f"_mesh{i}"
            ET.SubElement(root.find("asset"), "mesh", name=mesh_name, file=str(converted.resolve()),
                          scale=mesh.get("scale", "1 1 1"))
            ET.SubElement(body, "geom", name=name + f"_vis{i}", type="mesh", mesh=mesh_name,
                          group="1", contype="0", conaffinity="0", mass="0", rgba="0.35 0.38 0.42 1",
                          **pose(visual.find("origin")))
            # Convex hull of official collision shape for rigid chassis/frame.
            # Wheels use analytic rolling surfaces fitted to the CAD bounds.
            if link_name in ("left_wheel_link", "right_wheel_link"):
                extent = geometry.extents
                center = geometry.bounds.mean(axis=0)
                radius = max(extent[0], extent[2]) / 2
                ET.SubElement(body, "geom", name=name+"_tire", type="cylinder",
                              pos=" ".join(map(str, center)), quat="0.70710678 0.70710678 0 0",
                              size=f"{radius} {extent[1]/2}", friction="1.0 0.001 0.0001", condim="3", mass="0")
            elif link_name.endswith("wheel_link") and link_name[:2] in ("fl", "fr", "rl", "rr"):
                extent = geometry.extents
                # Mesh is in wheel frame; local Z is its rotation axis.
                ET.SubElement(body, "geom", name=name+"_roller", type="cylinder",
                              pos=" ".join(map(str, geometry.bounds.mean(axis=0))),
                              size=f"{max(extent[:2])/2} {extent[2]/2}", friction="0.8 0.001 0.0001", mass="0")
            elif link_name == "body_Link":
                # Separate CAD solids: a hull of the entire frame incorrectly
                # fills its open spaces. Omit thin sheets and small fasteners.
                pieces = geometry.split(only_watertight=False, repair=False)
                for k, piece in enumerate(pieces):
                    if min(piece.extents) < .003 or max(piece.extents) < .04:
                        continue
                    # Rotated zero-thickness sheets can have nonzero XYZ
                    # extents; reject them before MuJoCo's 3D hull compiler.
                    centered = piece.vertices - piece.vertices.mean(axis=0)
                    if np.linalg.svd(centered, compute_uv=False)[-1] / np.sqrt(len(centered)) < .0005:
                        continue
                    hull = piece.convex_hull
                    path = destination / f"mobile_frame_collision_{k}.obj"
                    hull.export(str(path))
                    mesh_id = f"mobile_frame_collision_{k}"
                    ET.SubElement(root.find("asset"), "mesh", name=mesh_id, file=str(path.resolve()))
                    ET.SubElement(body, "geom", name=mesh_id, type="mesh", mesh=mesh_id,
                                  mass="0", **pose(visual.find("origin")))
            else:
                for collision in link.findall("collision"):
                    ET.SubElement(body, "geom", name=name+f"_collision{i}", type="mesh", mesh=mesh_name,
                                  mass="0", **pose(collision.find("origin")))
        for joint in joints:
            if joint.find("parent").get("link") != link_name:
                continue
            child_name = joint.find("child").get("link")
            if child_name == "inertial_link":
                continue
            if child_name.startswith(("fl_base", "fr_base", "bl_base", "br_base")):
                continue
            child = add(child_name, body, joint.find("origin"))
            if child_name.endswith("wheel_link") and child_name[:2] in ("fl", "fr", "rl", "rr"):
                # Rollers sit inside chassis wheel wells; its convex hull has
                # no wheel-well cutouts. Other wheel contacts remain enabled.
                ET.SubElement(root.find("contact"), "exclude", body1="mobile_base_link", body2=child.get("name"))
            if joint.get("type") != "fixed":
                jname = "mobile_" + joint.get("name")
                ET.SubElement(child, "joint", name=jname, type="hinge", axis=joint.find("axis").get("xyz"),
                              damping="0.02", armature="0.01")
                if joint.get("name") in ("left_wheel", "right_wheel"):
                    ET.SubElement(root.find("actuator"), "velocity", name=jname+"_motor", joint=jname,
                                  kv="40", ctrllimited="true", ctrlrange="-6 6", forcelimited="true", forcerange="-20 20")
            ET.SubElement(root.find("contact"), "exclude", body1=name, body2=child.get("name"))
        return body

    for name in ("base_link", "body_Link"):
        joint = next(j for j in joints if j.find("child").get("link") == name)
        add(name, base, joint.find("origin"))
    radius = (bounds["left_wheel_link"][1][2] - bounds["left_wheel_link"][0][2]) / 2
    return {"wheel_radius": radius, "track_width": 0.34, "cad_bounds": bounds,
            "source": "aloha_new.urdf", "drive": "free root, two velocity-actuated wheels; passive caster hinges",
            "servo_calibrated": False, "drive_axle_z_correction_m": -0.011,
            "wheel_servo": {"kv": 40, "torque_limit_nm": 20, "speed_limit_rad_s": 6},
            "collision": "CAD chassis hull with wheel-well exclusions; frame component hulls (thickness under 3mm, span under 40mm and degenerate sheets omitted); CAD-fitted wheel cylinders"}
