"""Generate the MuJoCo asset for the tilting maze task.

The board is a cell-for-cell transcription of the supplied 32 x 32 workbook.
Each cell is emitted as either a level floor tile or a solid wall cell, so the
spreadsheet colors, collision geometry, and observations share one layout.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import numpy as np

from react_manip.cameras import look_at


def _vec(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _damping_ratio_for_restitution(restitution: float) -> float:
    """Map bounce strength to a stable MuJoCo contact damping ratio.

    The continuous spring formula becomes unstable here because the large
    ball can touch several adjacent wall boxes at once.  This interval was
    calibrated against the model's 2 ms timestep: it keeps repeated contacts
    stable while returning roughly 65--80 percent of the incoming speed.
    """
    restitution = min(0.88, max(0.65, float(restitution)))
    return min(0.14, max(0.10, 0.253 - 0.174 * restitution))


# Explicit 32 x 32 design grid transcribed from ``迷宫xlsx.xlsx`` (A1:AF32).
# ``#`` is the black outer retaining wall, ``r`` a red internal partition,
# ``t`` a yellow penalty tile, ``s`` the green start, ``G`` the blue target,
# and ``g``/``.`` traversable grey / white floor.  The pattern is kept as data
# so the renderer, collision geometry and scoring use one layout.
GRID_SIZE = 32


def _design_grid():
    # Cell classes come directly from the workbook's fills:
    # black border, red walls, green start, blue goal, grey/white floor, and
    # yellow penalty floor.  The outer border is intentionally one cell thick.
    return (
        "################################",
        "#gsgr........tttrrrrr..........#",
        "#gggr........tttr...r..........#",
        "#gggr........tttr...r..........#",
        "#gggr...rrrrr...r...r..........#",
        "#gggr...r.......r...rrrrrrrr...#",
        "#gggr...r.......rgggggggggggggg#",
        "#gggr...r.......rgggggggggggggg#",
        "#ggg.....ttt.....gggggggggggggg#",
        "#ggg.....ttt.....gggrrrr...rggg#",
        "#ggg.....ttt.....gggr......rggg#",
        "#gggr.......rrrrrgggr......rggg#",
        "#gggr.......r...rgggr...rrrrggg#",
        "#gggr.......r...rgggr...rgggggg#",
        "#gggrrrrr...r...rgggr...rgggggg#",
        "#gggggggggggr...rgggr...rgggggg#",
        "#gggggggggggr...rgggr...rgggttt#",
        "#gggggggggggr...rgggr...rgggttt#",
        "#...rrrrrgggr...rgggr...rgggttt#",
        "#...r...rgggr...rgggr...rgggrrr#",
        "#...r...rgggr...rggg.ttt.gggggg#",
        "#...r...rgggr...rggg.ttt.gggggg#",
        "#...r...rgggr...rggg.ttt.gggggg#",
        "#...r...rgggr...rgggr...rrrrggg#",
        "#...r...rgggr...rgggr......rggg#",
        "#.......rgggr...rgggr......rggg#",
        "#.......rgggr...rgggr......rggg#",
        "#.......rgggr...rgggrrrr...rggg#",
        "#tttr...rgggggggggggtttr...rGGG#",
        "#tttr...rgggggggggggtttr...rGGG#",
        "#tttr...rgggggggggggtttr...rGGG#",
        "################################",
    )


MAZE_GRID = _design_grid()


def config_from_scenario(parameters: dict) -> dict:
    """Normalize a sampled Scenario into the asset builder's small schema."""
    p = parameters or {}
    board = p.get("board", {})
    maze = p.get("maze", {})
    ball = p.get("ball", {})
    holes = p.get("holes", {})
    task = p.get("task", {})
    joystick = p.get("joystick", {})
    half = max(0.165, float(board.get("half_size_m", 0.17)))
    center_xy = tuple(float(v) for v in board.get("center_xy_m", (0.0, 0.0)))
    if len(center_xy) != 2 or not np.isfinite(center_xy).all():
        raise ValueError("board.center_xy_m must contain two finite coordinates")
    cell = 2 * half / GRID_SIZE
    start_cell = next((ix, iy) for iy, row in enumerate(MAZE_GRID)
                      for ix, kind in enumerate(row) if kind == "s")
    start_x, start_y = start_cell
    start_row = MAZE_GRID[start_y]
    left = start_x
    right = start_x
    while left > 0 and start_row[left - 1] not in "#r":
        left -= 1
    while right + 1 < GRID_SIZE and start_row[right + 1] not in "#r":
        right += 1
    rim_width = float(board.get("rim_width_m", 0.012))
    left_edge = max(-half + left * cell, -half + rim_width)
    right_edge = min(-half + (right + 1) * cell, half - rim_width)
    passage_width = right_edge - left_edge
    # Measure the actual opening between the inner rim and the red wall, not
    # one tile. It is about 30.5 mm wide at the default board scale.
    if "diameter_to_start_passage_ratio" in ball:
        radius = 0.5 * passage_width * float(ball["diameter_to_start_passage_ratio"])
    else:
        radius = min(0.4 * passage_width, float(ball.get("radius_m", 0.0122)))
    start_xy = list(ball.get("start_xy_m", (-0.14344, 0.14535)))
    if "start_x_m" in ball:
        start_xy[0] = ball["start_x_m"]
    if "start_y_m" in ball:
        start_xy[1] = ball["start_y_m"]
    clearance = 0.0004
    top_edge = min(half - start_y * cell, half - rim_width)
    # A 2.4-cell-diameter ball cannot have its center inside the one-cell
    # green tile beside the top wall. Start just below it, with the sphere
    # visibly overlapping the green tile and no initial wall penetration.
    start_xy = [float(np.clip(start_xy[0], left_edge + radius + clearance,
                              right_edge - radius - clearance)),
                float(min(start_xy[1], top_edge - radius - clearance))]
    def cell_xy(ch):
        return ((ch[1] + .5) * cell - half,
                half - (ch[0] + .5) * cell)
    trap_cells = [(x, y) for y, row in enumerate(MAZE_GRID)
                  for x, ch in enumerate(row) if ch == 't']
    goal_cells = [(x, y) for y, row in enumerate(MAZE_GRID)
                  for x, ch in enumerate(row) if ch == 'G']
    trap_xy = [cell_xy((x, y)) for x, y in trap_cells]
    goal_xy = tuple(np.mean(np.asarray([cell_xy((x, y)) for x, y in goal_cells]), axis=0))
    trap_half = cell * .48
    goal_half = cell * 1.45
    return {
        "board": {
            "half_size_m": half,
            "floor_height_m": float(board.get("floor_height_m", 0.018)),
            "rim_height_m": float(board.get("rim_height_m", 0.045)),
            "rim_width_m": float(board.get("rim_width_m", 0.012)),
            "center_height_m": float(board.get("center_height_m", 0.85)),
            "density_kg_m3": float(board.get("density_kg_m3", 80.0)),
            "surface_friction": float(board.get("surface_friction", 0.05)),
            "surface_rolling_friction": float(
                board.get("surface_rolling_friction", 0.000025)),
            "center_xy_m": center_xy,
        },
        "maze": {
            "wall_height_m": float(maze.get("wall_height_m", 0.034)),
            "wall_width_m": float(maze.get("wall_width_m", 0.0136)),
            "wall_friction": float(maze.get("wall_friction", 0.035)),
            "wall_rolling_friction": float(
                maze.get("wall_rolling_friction", 0.000015)),
            "wall_bounce": min(0.95, max(0.05, float(maze.get("wall_bounce", 0.78)))),
        },
        "ball": {
            "radius_m": radius,
            "mass_kg": min(0.013, float(ball.get("mass_kg", 0.011))),
            "friction": float(ball.get("friction", 0.01)),
            "rolling_friction": float(ball.get("rolling_friction", 0.000003)),
            "start_xy_m": tuple(float(v) for v in start_xy),
        },
        "holes": {
            "trap_xy_m": [tuple(float(v) for v in xy) for xy in trap_xy],
            "trap_half_size_m": trap_half,
            "goal_xy_m": goal_xy,
            "goal_half_size_m": goal_half,
        },
        "task": {
            "max_tilt_deg": min(6.0, max(0.1, float(task.get("max_tilt_deg", 6.0)))),
            "substeps_per_control": int(task.get("substeps_per_control", 10)),
            "horizon_sec": float(task.get("horizon_sec", 600.0)),
        },
        "joystick": {
            "max_angle_deg": float(joystick.get("max_angle_deg", 25.0)),
            "board_tilt_per_10_deg": min(4.0, max(
                1.0, float(joystick.get("board_tilt_per_10_deg", 2.5)))),
            "position_xy_m": tuple(joystick.get("position_xy_m", (-0.32, -0.20))),
            "contact_friction": float(joystick.get("contact_friction", 4.0)),
        },
    }


def _geom(parent, *, name, pos, size, rgba=None, material=None,
          friction=None, contype="1", conaffinity="1"):
    attrs = {"name": name, "type": "box", "pos": _vec(pos), "size": _vec(size),
             "contype": contype, "conaffinity": conaffinity}
    if rgba is not None:
        attrs["rgba"] = _vec(rgba)
    if material is not None:
        attrs["material"] = material
    if friction is not None:
        attrs["friction"] = _vec(friction)
    return ET.SubElement(parent, "geom", **attrs)


def make_xml(config: dict) -> str:
    b, m, ball, holes = (config[k] for k in ("board", "maze", "ball", "holes"))
    half = b["half_size_m"]
    root = ET.Element("mujoco", model="tilte_maze")
    ET.SubElement(root, "option", gravity="0 0 -9.81", integrator="implicitfast",
                  timestep="0.002", cone="elliptic")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="720", offheight="540")
    asset = ET.SubElement(root, "asset")
    for name, c1, c2 in (("maze_wood", "0.62 0.30 0.10", "0.90 0.63 0.28"),
                          ("maze_floor", "0.86 0.60 0.25", "0.98 0.82 0.48"),
                          ("maze_wall", "0.28 0.10 0.035", "0.57 0.25 0.07")):
        ET.SubElement(asset, "texture", name=name + "_tex", type="2d", builtin="checker",
                      width="256", height="256", rgb1=c1, rgb2=c2)
        ET.SubElement(asset, "material", name=name, texture=name + "_tex",
                      texrepeat="5 5" if name != "maze_floor" else "8 8",
                      texuniform="true", specular="0.22", shininess="0.28")
    ET.SubElement(asset, "material", name="maze_trim", rgba="0.015 0.018 0.022 1",
                  specular="0.16", shininess="0.22")
    # Flat materials for the small tray. High-frequency checker textures on
    # the floor and walls shimmer when the camera or board changes angle.
    ET.SubElement(asset, "material", name="maze_floor_flat", rgba="0.82 0.56 0.20 1",
                  specular="0.08", shininess="0.16")
    ET.SubElement(asset, "material", name="maze_wall_black", rgba="0.015 0.018 0.022 1",
                  specular="0.10", shininess="0.18")
    ET.SubElement(asset, "material", name="maze_wall_red", rgba="0.78 0.035 0.035 1",
                  specular="0.10", shininess="0.18")
    ET.SubElement(asset, "material", name="maze_floor_white", rgba="0.93 0.93 0.93 1")
    ET.SubElement(asset, "material", name="maze_floor_gray", rgba="0.62 0.62 0.64 1")
    ET.SubElement(asset, "material", name="maze_floor_trap", rgba="0.95 0.78 0.16 1")
    ET.SubElement(asset, "material", name="maze_floor_start", rgba="0.18 0.64 0.20 1")
    ET.SubElement(asset, "material", name="maze_floor_goal", rgba="0.05 0.38 0.82 1")
    ET.SubElement(asset, "material", name="maze_ball_bright",
                  rgba="1 0.08 0.75 1", emission="0.35", specular="0.4", shininess="0.5")
    world = ET.SubElement(root, "worldbody")
    # Table and room context, matching the room/lighting convention of the
    # existing laptop task while keeping the maze close to the robot workspace.
    _geom(world, name="table", pos=(0, 0, 0.75), size=(1.45, 1.25, 0.05),
          rgba=(0.46, 0.29, 0.15, 1), friction=(0.7, 0.02, 0.001),
          contype="2", conaffinity="2")
    # Make the support table explicit in overview views. Legs are visual-only
    # so they cannot interfere with the maze or a future arm base.
    for i, (x, y) in enumerate(((-1.18, -0.98), (-1.18, 0.98),
                                 (1.18, -0.98), (1.18, 0.98))):
        _geom(world, name=f"table_leg_{i}", pos=(x, y, 0.35),
              size=(0.055, 0.055, 0.35), rgba=(0.34, 0.19, 0.09, 1),
              contype="0", conaffinity="0")
    # One tabletop mount, a visible ball socket under the tray, and a physical
    # two-axis joystick in front of the arm. The socket does not contact the
    # moving tray; the ball joint below supplies the exact pivot constraint.
    _geom(world, name="arm_mount", pos=(-0.75, -0.20, 0.808),
          size=(0.12, 0.12, 0.008), rgba=(0.16, 0.18, 0.20, 0),
          friction=(0.8, 0.01, 0.001), contype="2", conaffinity="2")
    board_x, board_y = b["center_xy_m"]
    ET.SubElement(world, "geom", name="maze_socket", type="cylinder",
                  pos=_vec((board_x, board_y, 0.825)), size="0.025 0.025", rgba="0.28 0.29 0.31 1",
                  contype="0", conaffinity="0")
    ET.SubElement(world, "geom", name="maze_ball_pivot", type="sphere",
                  pos=_vec((board_x, board_y, b["center_height_m"])), size="0.012",
                  rgba="0.75 0.77 0.79 1", contype="0", conaffinity="0")
    stick_x, stick_y = config["joystick"]["position_xy_m"]
    ET.SubElement(world, "geom", name="joystick_base", type="cylinder",
                  pos=_vec((stick_x, stick_y, 0.807)), size="0.052 0.007",
                  rgba="0.19 0.21 0.24 1", contype="0", conaffinity="0")
    stick = ET.SubElement(world, "body", name="joystick",
                          pos=_vec((stick_x, stick_y, 0.815)))
    ET.SubElement(stick, "joint", name="joystick_ball", type="ball",
                  limited="true", range=f"0 {config['joystick']['max_angle_deg']}",
                  damping="0.04", stiffness="0.4")
    ET.SubElement(stick, "geom", name="joystick_stem", type="cylinder",
                  pos="0 0 0.075", size="0.008 0.075", mass="0.025",
                  rgba="0.72 0.74 0.77 1",
                  friction=_vec((config["joystick"]["contact_friction"] * .75, .03, .003)))
    ET.SubElement(stick, "geom", name="joystick_grip", type="sphere",
                  pos="0 0 0.155", size="0.030", mass="0.035",
                  rgba="0.13 0.34 0.75 1",
                  friction=_vec((config["joystick"]["contact_friction"], .03, .003)))
    _geom(world, name="room_floor", pos=(0, 0, -0.04), size=(3, 3, 0.04),
          rgba=(0.65, 0.68, 0.70, 1), contype="0", conaffinity="0")
    for name, pos, size in (("room_back", (0, 2.8, 1.5), (3, .05, 1.5)),
                            ("room_left", (-2.8, 0, 1.5), (.05, 3, 1.5)),
                            ("room_right", (2.8, 0, 1.5), (.05, 3, 1.5))):
        _geom(world, name=name, pos=pos, size=size, rgba=(0.86, 0.87, 0.89, 1),
              contype="0", conaffinity="0")

    board = ET.SubElement(world, "body", name="maze_board",
                          pos=_vec((board_x, board_y, b["center_height_m"])))
    ET.SubElement(board, "joint", name="board_ball", type="ball",
                  limited="true", range=f"0 {config['task']['max_tilt_deg']}",
                  damping="0.08")
    floor_z = b["floor_height_m"] / 2
    # One unbroken contact floor prevents the small sphere from catching on
    # numerical seams between the 32 x 32 color tiles. The visible layout and
    # all red/black wall collision cells remain exactly as authored.
    floor_contact = _geom(board, name="board_floor_collision",
                          pos=(0, 0, floor_z),
                          size=(half, half, b["floor_height_m"] / 2),
                          rgba=(0, 0, 0, 0),
                          friction=(b["surface_friction"], 0.001,
                                    b["surface_rolling_friction"]))
    floor_contact.set("condim", "6")
    floor_contact.set("priority", "1")
    # Build exactly the 32 x 32 spreadsheet cells. Black outer and red
    # internal cells are physical partitions; all other colored cells remain
    # level floor tiles.
    grid = GRID_SIZE
    tile = (2 * half) / grid
    for ix in range(grid):
        for iy in range(grid):
            x = -half + (ix + .5) * tile
            y = half - (iy + .5) * tile
            cell_kind = MAZE_GRID[iy][ix]
            if cell_kind in ('#', 'r'):
                continue
            mat = {'.': "maze_floor_white", 'g': "maze_floor_gray",
                   't': "maze_floor_trap", 's': "maze_floor_start",
                   'G': "maze_floor_goal"}.get(cell_kind, "maze_floor_white")
            _geom(board, name=f"board_floor_{ix}_{iy}", pos=(x, y, floor_z),
                  size=(tile / 2, tile / 2, b["floor_height_m"] / 2), material=mat,
                  contype="0", conaffinity="0").set("mass", "0")
    rim_h, rim_w = b["rim_height_m"], b["rim_width_m"]
    for name, pos, size in (("rim_left", (-half, 0, rim_h / 2), (rim_w, half + rim_w, rim_h / 2)),
                            ("rim_right", (half, 0, rim_h / 2), (rim_w, half + rim_w, rim_h / 2)),
                            ("rim_front", (0, -half, rim_h / 2), (half + rim_w, rim_w, rim_h / 2)),
                            ("rim_back", (0, half, rim_h / 2), (half + rim_w, rim_w, rim_h / 2))):
        # Use a single matte trim material on the thin rim.  A checker texture
        # on this narrow, high-contrast edge produced distracting aliasing and
        # made the border appear to flicker when the board tilted.
        _geom(board, name=name, pos=pos, size=size, material="maze_trim",
              friction=(m["wall_friction"], 0.001, m["wall_rolling_friction"]))

    # Wall grid cells are raised solid partitions. A tiny bevel-sized gap is
    # avoided: adjacent cells touch, so the ball cannot leak through walls.
    wi = 0
    for iy, row in enumerate(MAZE_GRID):
        for ix, cell_kind in enumerate(row):
            if cell_kind not in ('#', 'r'):
                continue
            x = -half + (ix + .5) * tile
            y = half - (iy + .5) * tile
            wall_mat = "maze_wall_black" if cell_kind == '#' else "maze_wall_red"
            _geom(board, name=f"maze_wall_{wi}", pos=(x, y, m["wall_height_m"] / 2),
              size=(tile / 2, tile / 2, m["wall_height_m"] / 2), material=wall_mat,
              friction=(m["wall_friction"], 0.001, m["wall_rolling_friction"]))
            wi += 1

    # The tray is a lightweight plastic/wood shell, not a solid 1,000 kg/m^3
    # block. Apply one explicit density to every board collision primitive.
    for geom in board.findall("geom"):
        if geom.get("mass") is None:
            geom.set("density", str(b["density_kg_m3"]))
        geom.set("margin", "0")
        name = geom.get("name", "")
        if name.startswith("maze_wall_") or name.startswith("rim_"):
            damping = _damping_ratio_for_restitution(m["wall_bounce"])
            geom.set("priority", "2")
            geom.set("solref", _vec((0.01, damping)))
        else:
            geom.set("solref", "0.02 1")
        geom.set("solimp", "0.95 0.99 0.001")
    floor_contact.set("solref", "0.008 1")

    # MuJoCo free joints are only valid on top-level bodies.  Keep the ball at
    # world scope while the board rotates underneath it; physical contacts
    # with the board floor and maze walls are preserved.
    ball_body = ET.SubElement(world, "body", name="ball",
                              pos=_vec((board_x + ball["start_xy_m"][0],
                                         board_y + ball["start_xy_m"][1],
                                         b["center_height_m"] +
                                         b["floor_height_m"] + ball["radius_m"] + .003)))
    ET.SubElement(ball_body, "joint", name="ball_free", type="free")
    ET.SubElement(ball_body, "geom", name="ball_geom", type="sphere", size=str(ball["radius_m"]),
                  mass=str(ball["mass_kg"]),
                  friction=_vec((ball["friction"], .0005, ball["rolling_friction"])),
                  material="maze_ball_bright", condim="6", margin="0",
                  solref="0.008 1", solimp="0.95 0.99 0.001")
    def camera(name, position, target, fovy, up=(0., 0., 1.)):
        rot = look_at(np.asarray(position, float), np.asarray(target, float), up=up)
        ET.SubElement(world, "camera", name=name, pos=_vec(position),
                      xyaxes=_vec(np.r_[rot[:, 0], rot[:, 1]]), fovy=str(fovy))
    # The enlarged tray needs a wider closeup without losing the ball.
    camera("workspace_oblique", (-1.05, -1.60, 1.28),
           (board_x - 0.02, board_y + 0.02, 0.88), 42)
    ET.SubElement(world, "camera", name="workspace_top",
                  pos=_vec((board_x, board_y, 1.55)),
                  xyaxes="1 0 0 0 1 0", fovy="52")
    ET.SubElement(world, "camera", name="maze_closeup",
                  pos=_vec((board_x, board_y, 1.42)),
                  xyaxes="1 0 0 0 1 0", fovy="44")
    camera("joystick_closeup", (-0.48, -0.52, 1.20), (stick_x, stick_y, 0.90), 40)
    camera("room_overview", (3.0, -4.8, 3.1),
           ((board_x + stick_x) / 2, (board_y + stick_y) / 2, 0.72), 68)
    ET.SubElement(world, "light", name="ceiling_light", pos="0 0 3", dir="0 0 -1",
                  diffuse="0.8 0.8 0.8", ambient="0.25 0.25 0.25")
    return ET.tostring(root, encoding="unicode")
