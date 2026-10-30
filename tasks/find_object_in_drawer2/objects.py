"""Small, recognizable procedural objects with matching physical geometry.

Dimensions are in metres. No external downloads or class-colour mapping is
needed: silhouettes and parts identify 36 unique household objects. Small objects
use graspable benchmark-scale dimensions rather than life-size replicas.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

# Stable catalog IDs. Every item has its own silhouette and language name.
CATALOG = (
    ("cup", "杯子"), ("bottle", "瓶子"), ("apple", "苹果"),
    ("book", "书本"), ("toy_car", "玩具车"), ("stapler", "订书机"),
    ("pen", "笔"), ("eraser", "橡皮"), ("ruler", "尺子"),
    ("scissors", "剪刀"), ("tape_roll", "胶带卷"), ("calculator", "计算器"),
    ("phone", "手机"), ("remote_control", "遥控器"), ("mouse", "鼠标"),
    ("camera", "相机"), ("flashlight", "手电筒"), ("alarm_clock", "闹钟"),
    ("hammer", "锤子"), ("screwdriver", "螺丝刀"), ("wrench", "扳手"),
    ("pliers", "钳子"), ("padlock", "挂锁"), ("key", "钥匙"),
    ("comb", "梳子"), ("toothbrush", "牙刷"), ("soap", "肥皂"),
    ("sponge", "海绵"), ("glasses_case", "眼镜盒"), ("brush", "刷子"),
    ("banana", "香蕉"), ("lemon", "柠檬"), ("pear", "梨"),
    ("bread", "面包"), ("donut", "甜甜圈"), ("milk_carton", "牛奶盒"),
)
OBJECT_NAMES = tuple(name for name, _ in CATALOG)
NAMES_ZH = tuple(zh for _, zh in CATALOG)
NAMES_EN = tuple(name.replace("_", " ") for name in OBJECT_NAMES)
LABELS = {name: (name.replace("_", " "), zh) for name, zh in CATALOG}
# Body-local pinch centres. Handles / narrow waists are preferred over AABB
# centres (which can lie in the empty space of tools). Metres; closing axis Y.
GRASP_POINTS = {
    "cup": (0, 0, .030), "bottle": (0, 0, .030), "apple": (0, 0, .027),
    "pen": (0, 0, .010), "scissors": (-.020, 0, .014),
    "hammer": (-.014, 0, .012), "screwdriver": (-.022, 0, .014),
    "wrench": (0, 0, .012), "pliers": (-.020, 0, .013),
    "key": (-.022, 0, .012), "toothbrush": (-.018, 0, .010),
    "brush": (-.025, 0, .010), "banana": (0, .008, .018),
    "pear": (0, 0, .025), "lemon": (0, 0, .023),
    "flashlight": (-.010, 0, .014), "camera": (-.010, 0, .021),
}


def vec(values):
    return " ".join(f"{float(x):.8g}" for x in values)


def add_object(world, name):
    """Build a free body with its bottom at local z=0; return its height."""
    body = ET.SubElement(world, "body", name=name, pos="0 0 0.7")
    ET.SubElement(body, "freejoint", name=f"{name}_free")
    part_index = 0

    def part(kind, size, pos, color, euler=None):
        nonlocal part_index
        attrs = dict(type=kind, size=vec(size), pos=vec(pos), rgba=vec((*color, 1)),
                     friction="0.8 0.01 0.001", condim="6", solref="0.005 1",
                     density="500")
        if euler is not None:
            attrs["euler"] = vec(euler)
        for visual in (False, True):
            ET.SubElement(body, "geom", name=f"{name}_{part_index}" + ("_vis" if visual else ""),
                          group="1" if visual else "0", contype="0" if visual else "1",
                          conaffinity="0" if visual else "1", **{**attrs, **({"density": "0"} if visual else {})})
        part_index += 1

    if name == "cup":
        blue = (.18, .48, .72)
        # Inset flat foot avoids a cylinder rim rocking between contact points.
        part("box", (.016, .016, .001), (0, 0, .001), blue)
        part("cylinder", (.024, .0025), (0, 0, .0035), blue)
        for i in range(12):
            angle = i * math.pi / 6
            part("box", (.003, .0068, .025),
                 (.023 * math.cos(angle), .023 * math.sin(angle), .030), blue, (0, 0, angle))
        for pos, size in [((.033, 0, .045), (.010, .005, .004)),
                          ((.033, 0, .015), (.010, .005, .004)),
                          ((.042, 0, .030), (.004, .005, .018))]:
            part("box", size, pos, blue)
        return .055
    if name == "bottle":
        part("box", (.015, .015, .001), (0, 0, .001), (.25, .65, .40))
        part("cylinder", (.023, .0275), (0, 0, .0285), (.25, .65, .40))
        part("cylinder", (.012, .007), (0, 0, .063), (.25, .65, .40))
        part("cylinder", (.013, .004), (0, 0, .074), (.88, .88, .86))
        return .078
    if name == "apple":
        part("sphere", (.027,), (0, 0, .027), (.75, .13, .09))
        part("cylinder", (.003, .006), (0, 0, .056), (.25, .13, .05))
        return .062
    if name == "book":
        part("box", (.040, .026, .010), (0, 0, .013), (.91, .88, .77))
        for z in (.0015, .0245):
            part("box", (.043, .029, .0015), (0, 0, z), (.55, .20, .16))
        part("box", (.043, .002, .013), (0, -.028, .013), (.55, .20, .16))
        return .026
    if name == "toy_car":
        part("box", (.040, .022, .011), (0, 0, .023), (.91, .64, .10))
        part("box", (.020, .020, .009), (-.005, 0, .043), (.22, .35, .43))
        for x in (-.026, .026):
            for y in (-.026, .026):
                part("cylinder", (.012, .005), (x, y, .012), (.08, .08, .09), (math.pi / 2, 0, 0))
        return .052
    if name == "stapler":
        part("box", (.042, .019, .004), (0, 0, .004), (.22, .23, .25))
        part("box", (.008, .019, .013), (.030, 0, .021), (.32, .33, .35))
        part("box", (.042, .019, .006), (0, 0, .034), (.22, .23, .25))
        return .040
    # Shared physical construction helpers; all details also have collisions.
    metal, dark, ivory = (.60, .64, .68), (.13, .16, .20), (.90, .88, .78)
    blue, red, wood = (.18, .43, .68), (.75, .20, .16), (.55, .32, .15)

    def box(size, pos, color, yaw=0.):
        part("box", size, pos, color, (0, 0, yaw))

    def cylinder(radius, half_length, pos, color, along_x=False):
        part("cylinder", (radius, half_length), pos, color,
             (0, math.pi/2, 0) if along_x else None)

    def ring(cx, cy, z, outer, inner, half_height, color):
        # Faceted annulus with a real opening, not a filled cylinder.
        radius = (outer + inner) / 2
        for k in range(16):
            angle = 2 * math.pi * k / 16
            box(((outer-inner)/2, radius*math.tan(math.pi/16), half_height),
                (cx+radius*math.cos(angle), cy+radius*math.sin(angle), z), color, angle)

    if name == "pen":
        cylinder(.010, .030, (0,0,.010), blue, True)
        cylinder(.007, .009, (.037,0,.010), metal, True)
        box((.018,.003,.002),(-.010,0,.021),metal)
        return .023
    if name == "eraser":
        box((.024,.018,.012),(0,0,.012),(.91,.47,.49))
        box((.010,.0185,.0125),(-.002,0,.0125),ivory)
        return .025
    if name == "ruler":
        box((.043,.013,.008),(0,0,.008),(.89,.70,.23))
        for k in range(9):
            box((.0007,.006 if k%2==0 else .003,.0005),(-.036+.009*k,-.005,.0165),dark)
        return .017
    if name == "scissors":
        for y in (-.016,.016):
            ring(-.022,y,.013,.014,.007,.010,red)
        box((.025,.004,.004),(.018,-.006,.014),metal,-.25)
        box((.025,.004,.004),(.018,.006,.014),metal,.25)
        cylinder(.005,.006,(-.003,0,.014),dark)
        return .023
    if name == "tape_roll":
        ring(0,0,.018,.027,.016,.018,(.78,.63,.38))
        ring(0,0,.018,.017,.014,.017,ivory)
        return .036
    if name == "calculator":
        box((.038,.025,.010),(0,0,.010),dark)
        box((.010,.020,.001),(.023,0,.021),(.40,.57,.48))
        for x in (-.025,-.013,-.001):
            for y in (-.015,0,.015):box((.004,.004,.002),(x,y,.022),ivory)
        return .024
    if name == "phone":
        box((.041,.024,.008),(0,0,.008),dark)
        box((.034,.020,.001),(.002,0,.017),(.18,.53,.75))
        cylinder(.003,.001,(-.037,0,.017),ivory)
        return .018
    if name == "remote_control":
        box((.041,.016,.011),(0,0,.011),dark)
        cylinder(.004,.002,(.028,0,.023),red)
        ring(.010,0,.024,.008,.003,.002,metal)
        for x in (-.010,-.023):
            for y in (-.008,.008):box((.003,.003,.002),(x,y,.024),ivory)
        return .026
    if name == "mouse":
        box((.025,.020,.006),(0,0,.006),dark)
        part("ellipsoid",(.033,.023,.014),(0,0,.014),(.55,.58,.62))
        box((.009,.001,.001),(.021,0,.027),dark)
        cylinder(.003,.005,(.012,0,.029),dark,True)
        return .032
    if name == "camera":
        box((.030,.024,.020),(0,0,.020),dark)
        cylinder(.017,.008,(.008,0,.048),metal)
        cylinder(.013,.001,(.008,0,.057),(.13,.28,.39))
        box((.010,.009,.004),(-.016,0,.044),dark)
        return .058
    if name == "flashlight":
        cylinder(.013,.026,(-.010,0,.014),dark,True)
        cylinder(.020,.010,(.026,0,.020),metal,True)
        cylinder(.016,.001,(.037,0,.020),(.95,.88,.52),True)
        return .040
    if name == "alarm_clock":
        box((.015,.021,.006),(0,0,.006),dark)
        cylinder(.025,.014,(0,0,.017),(.82,.26,.20))
        cylinder(.022,.001,(0,0,.032),ivory)
        box((.012,.001,.001),(.010,0,.034),dark,.4)
        box((.001,.010,.001),(0,.008,.034),dark)
        for y in (-.023,.023):part("sphere",(.009,),(-.017,y,.028),metal)
        return .037
    if name == "hammer":
        box((.031,.010,.011),(-.009,0,.011),wood)
        box((.010,.028,.015),(.025,0,.015),metal)
        return .030
    if name == "screwdriver":
        cylinder(.014,.019,(-.023,0,.014),red,True)
        cylinder(.004,.018,(.014,0,.014),metal,True)
        box((.006,.006,.002),(.037,0,.014),metal)
        return .028
    if name == "wrench":
        box((.026,.011,.010),(0,0,.010),metal)
        for x in (-.032,.032):
            box((.009,.008,.010),(x,-.014,.010),metal)
            box((.009,.008,.010),(x,.014,.010),metal)
        return .020
    if name == "pliers":
        for y in (-1,1):
            box((.024,.007,.011),(-.016,y*.014,.011),red,y*.15)
            box((.015,.006,.008),(.024,y*.009,.012),metal,-y*.20)
        cylinder(.009,.003,(.004,0,.022),metal)
        return .025
    if name == "padlock":
        box((.023,.019,.014),(-.012,0,.014),(.77,.59,.19))
        ring(.017,0,.012,.022,.016,.007,metal)
        box((.004,.002,.0005),(-.012,0,.0285),dark)
        return .029
    if name == "key":
        ring(-.025,0,.012,.016,.007,.012,(.77,.63,.24))
        box((.025,.006,.005),(.012,0,.012),(.77,.63,.24))
        for x in (.018,.030):box((.004,.007,.005),(x,.008,.012),(.77,.63,.24))
        return .024
    if name == "comb":
        box((.040,.007,.010),(0,-.010,.010),(.52,.22,.43))
        for k in range(10):box((.002,.016,.007),(-.036+.008*k,.007,.010),(.52,.22,.43))
        return .020
    if name == "toothbrush":
        box((.031,.009,.010),(-.010,0,.010),blue)
        box((.012,.012,.007),(.028,0,.012),ivory)
        for x in (.020,.028,.036):
            for y in (-.007,0,.007):box((.0025,.0025,.005),(x,y,.023),ivory)
        return .028
    if name == "soap":
        box((.027,.018,.008),(0,0,.008),(.71,.86,.60))
        part("ellipsoid",(.032,.022,.010),(0,0,.014),(.71,.86,.60))
        box((.010,.006,.001),(0,0,.024),(.44,.64,.38))
        return .025
    if name == "sponge":
        box((.031,.023,.013),(0,0,.013),(.92,.77,.20))
        box((.031,.023,.003),(0,0,.029),(.22,.43,.21))
        return .032
    if name == "glasses_case":
        box((.036,.018,.007),(0,0,.007),(.27,.18,.37))
        part("ellipsoid",(.042,.022,.011),(0,0,.016),(.27,.18,.37))
        box((.032,.0008,.001),(0,-.021,.014),metal)
        return .027
    if name == "brush":
        box((.018,.010,.010),(-.023,0,.010),wood)
        box((.020,.022,.008),(.013,0,.010),wood)
        for x in (0,.010,.020,.030):
            for y in (-.015,-.005,.005,.015):cylinder(.003,.008,(x,y,.026),dark)
        return .034
    if name == "banana":
        yellow=(.94,.77,.12)
        # Use overlapping ellipsoids instead of a very flat row of spheres.
        # The previous 23 mm tall banana was easy to skim over with the
        # gripper fingers, especially when it rested directly on the drawer
        # floor.  A slightly thicker, continuous body gives the fingers a
        # reliable contact patch while remaining well within the 110 mm
        # drawer clearance.
        for k in range(7):
            x=-.030+.010*k; y=.010-.020*(x/.030)**2
            z=.015 + .003 * (1.0 - (x/.030)**2)
            part("ellipsoid",(.011,.013,.015),(x,y,z),yellow)
        box((.007,.004,.004),(-.037,-.010,.013),wood,-.5)
        box((.012,.006,.001),(0,.007,.001),yellow)
        return .033
    if name == "lemon":
        part("ellipsoid",(.034,.023,.022),(0,0,.023),(.94,.83,.13))
        box((.013,.011,.001),(0,0,.001),(.94,.83,.13))
        for x in (-.034,.034):part("sphere",(.005,),(x,0,.023),(.94,.83,.13))
        return .045
    if name == "pear":
        part("sphere",(.024,),(0,0,.025),(.64,.76,.25))
        part("ellipsoid",(.015,.015,.019),(.003,0,.046),(.64,.76,.25))
        cylinder(.0025,.005,(.003,0,.068),wood)
        box((.012,.012,.001),(0,0,.001),(.64,.76,.25))
        return .073
    if name == "bread":
        box((.030,.022,.014),(0,0,.014),(.66,.36,.13))
        part("ellipsoid",(.035,.024,.013),(0,0,.026),(.78,.52,.25))
        for x in (-.017,0,.017):box((.002,.017,.0006),(x,0,.039),ivory,.35)
        return .0396
    if name == "donut":
        ring(0,0,.012,.029,.012,.012,(.72,.40,.17))
        ring(0,0,.025,.027,.013,.002,(.83,.42,.50))
        return .027
    if name == "milk_carton":
        box((.019,.017,.027),(0,0,.027),ivory)
        box((.0195,.0175,.009),(0,0,.026),blue)
        for x in (-1,1):
            part("box",(.013,.017,.003),(x*.009,0,.062),ivory,(0,x*math.pi/4,0))
        box((.002,.017,.004),(0,0,.071),blue)
        return .075
    raise ValueError(f"unknown object: {name}")
