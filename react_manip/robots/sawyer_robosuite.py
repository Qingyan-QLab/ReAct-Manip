"""Sawyer model with its physical pedestal integrated into the robot asset.

The upstream robosuite Sawyer is normally composed with ``RethinkMount``.
That mount is an environment-time support model whose root is offset below
the arm by roughly 0.912 m.  This is convenient for table arenas, but makes a
stand-alone robot appear to float when the environment does not provide that
mount.  ``SawyerWithBase`` keeps the arm at the calibrated height and adds a
simple collision/visual pedestal to the Sawyer MJCF itself.  It is therefore
part of the robot embodiment and follows the robot whenever its root is
placed or moved.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


# Height of the upstream Rethink pedestal from its floor contact to the
# Sawyer arm root.  The original RethinkMount XML has a child root at -0.922 m
# and a top offset of -0.01 m, yielding -0.912 m as the model bottom offset.
SAWYER_ARM_ROOT_Z = 0.912
SAWYER_BASE_RADIUS = 0.31
SAWYER_BASE_HEIGHT = SAWYER_ARM_ROOT_Z


def register_sawyer_with_base() -> str:
    """Register and return the robosuite name for the integrated-base Sawyer.

    Imports robosuite lazily so task discovery and schema checks remain usable
    without simulator dependencies.
    """

    from robosuite.models.robots.manipulators.sawyer_robot import Sawyer
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot
    from robosuite.utils.mjcf_utils import find_elements

    name = "CLSawyer"
    if name not in REGISTERED_ROBOTS:

        class CLSawyer(Sawyer):
            """Sawyer with a floor-contacting pedestal in its own MJCF."""

            def __init__(self, idn=0):
                super().__init__(idn=idn)

                # The arm link in robosuite's Sawyer XML is rooted at z=0 and
                # relies on RethinkMount to lift it.  Move only the arm branch;
                # the robot root remains at the floor for normal placement.
                arm_root = find_elements(
                    root=self.worldbody,
                    tags="body",
                    attribs={"name": self.correct_naming("right_arm_base_link")},
                    return_first=True,
                )
                if arm_root is None:
                    raise ValueError("Sawyer asset is missing right_arm_base_link")
                arm_root.set("pos", f"0 0 {SAWYER_ARM_ROOT_Z:.6f}")

                # The upstream robotview is attached directly to the XML
                # root at z=0.4 because RethinkMount lifts that root by
                # 0.912 m.  With the integrated pedestal the root is on the
                # floor, so preserve the original world camera height.
                robotview = find_elements(
                    root=self.worldbody,
                    tags="camera",
                    attribs={"name": self.correct_naming("robotview")},
                    return_first=True,
                )
                if robotview is not None:
                    robotview.set("pos", f"1.0 0 {0.4 + SAWYER_ARM_ROOT_Z:.6f}")

                root = self._elements["root_body"]
                base = ET.Element(
                    "body",
                    {
                        "name": self.correct_naming("sawyer_integrated_base"),
                        "pos": "0 0 0",
                    },
                )
                # Static root-body geometry is physically attached to the
                # robot and therefore moves with it, while the visual copy is
                # excluded from contacts to avoid duplicate collision forces.
                ET.SubElement(
                    base,
                    "geom",
                    {
                        "name": self.correct_naming("sawyer_base_collision"),
                        "type": "cylinder",
                        "size": f"{SAWYER_BASE_RADIUS:.4f} {SAWYER_BASE_HEIGHT / 2:.4f}",
                        "pos": f"0 0 {SAWYER_BASE_HEIGHT / 2:.4f}",
                        "rgba": "0.12 0.14 0.16 1",
                        "density": "450",
                        "friction": "0.8 0.01 0.001",
                        "condim": "6",
                    },
                )
                ET.SubElement(
                    base,
                    "geom",
                    {
                        "name": self.correct_naming("sawyer_base_visual"),
                        "type": "cylinder",
                        # Keep the underside exactly on the floor; only the
                        # radius is enlarged for a small visual lip.
                        "size": f"{SAWYER_BASE_RADIUS + 0.006:.4f} {SAWYER_BASE_HEIGHT / 2:.4f}",
                        "pos": f"0 0 {SAWYER_BASE_HEIGHT / 2:.4f}",
                        "rgba": "0.18 0.20 0.23 1",
                        "contype": "0",
                        "conaffinity": "0",
                        "group": "1",
                    },
                )
                root.append(base)

            @property
            def default_base(self):
                # NullMount contributes no environment-specific pedestal or
                # vertical offset.  The pedestal above is the robot asset.
                return "NullMount"

            @property
            def top_offset(self):
                # Preserve the upstream top envelope (1.0 m above the arm
                # root) while the arm root itself is lifted by the integrated
                # pedestal.
                return np.array((0.0, 0.0, SAWYER_ARM_ROOT_Z + 1.0))

            @property
            def _horizontal_radius(self):
                return max(float(super()._horizontal_radius), SAWYER_BASE_RADIUS)

        # RobotModelMeta registers the class in REGISTERED_ROBOTS.  The
        # wrapper map is what RobotEnv uses when creating the runtime robot.
        ROBOT_CLASS_MAPPING[name] = FixedBaseRobot
        # Keep the public robosuite robot id stable.  Existing task code often
        # passes ``robots="Sawyer"`` directly; that path must receive the same
        # integrated-base embodiment as the benchmark factory's ``sawyer`` ID.
        REGISTERED_ROBOTS["Sawyer"] = CLSawyer
        ROBOT_CLASS_MAPPING["Sawyer"] = FixedBaseRobot

    return name


__all__ = ["register_sawyer_with_base", "SAWYER_ARM_ROOT_Z"]
