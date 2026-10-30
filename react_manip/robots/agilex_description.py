"""Legacy extractor for the archived mobile_aloha_sim follower model.

This prepares URDF input; it does not implement a runnable task or convert DAE
meshes. Source: agilexrobotics/mobile_aloha_sim, aloha_new.urdf.
"""
from copy import deepcopy
import xml.etree.ElementTree as ET


def extract_followers(source: ET.Element, embodiment: str) -> ET.Element:
    """Keep front follower subtrees, excluding leaders, chassis and cameras.

    Single PiPER is rooted at its base origin. Dual-arm mounting transforms are
    preserved relative to the upstream body_Link frame (not the world frame).
    This legacy model is not the official PiPER-X asset.
    """
    if embodiment in ("piperx", "dual_piperx", "mobile_dual_piperx"):
        raise ValueError("PiPER-X requires scripts/build_agilex_assets.py and official agx_arm_urdf assets")
    selections = {"piper": ("fr",), "agilex_aloha": ("fl", "fr")}
    if embodiment not in selections:
        raise ValueError("embodiment must be piper or agilex_aloha")
    links = {link.attrib["name"]: link for link in source.findall("link")}
    joints = source.findall("joint")
    output = ET.Element("robot", name=embodiment + "_followers")
    ET.SubElement(output, "link", name="world")
    for material in source.findall("material"):
        output.append(deepcopy(material))
    for prefix in selections[embodiment]:
        base = prefix + "_base_link"
        if base not in links:
            raise ValueError(f"missing follower base: {base}")
        selected = {base}
        chosen = []
        pending = list(joints)
        while True:
            additions = [j for j in pending if j.find("parent").get("link") in selected]
            if not additions:
                break
            for joint in additions:
                child = joint.find("child").get("link")
                if child in selected or child not in links:
                    raise ValueError(f"invalid follower tree at {child}")
                selected.add(child)
                chosen.append(joint)
                pending.remove(joint)
        revolute = [j for j in chosen if j.get("type") == "revolute"]
        fingers = [j for j in chosen if j.get("type") == "prismatic"]
        if len(revolute) != 6 or len(fingers) != 2:
            raise ValueError(f"{prefix}: expected six arm and two finger joints")
        for joint in revolute + fingers:
            limit = joint.find("limit")
            if limit is None or not {"lower", "upper", "effort", "velocity"} <= set(limit.keys()):
                raise ValueError(f"missing limits: {joint.get('name')}")
            child = links[joint.find("child").get("link")]
            if child.find("inertial") is None or child.find("collision") is None:
                raise ValueError(f"missing dynamics/collision: {child.get('name')}")
        mounts = [j for j in joints if j.find("child").get("link") == base]
        if len(mounts) != 1 or mounts[0].get("type") != "fixed":
            raise ValueError(f"expected fixed mount for {base}")
        mount = deepcopy(mounts[0])
        if mount.find("parent").get("link") != "body_Link":
            raise ValueError("unexpected mounting frame; inspect upstream model")
        mount.find("parent").set("link", "world")
        if embodiment == "piper":
            origin = mount.find("origin")
            if origin is None:
                origin = ET.SubElement(mount, "origin")
            origin.set("xyz", "0 0 0")
            origin.set("rpy", "0 0 0")
        for name in sorted(selected):
            output.append(deepcopy(links[name]))
        output.append(mount)
        output.extend(deepcopy(chosen))
    return output
