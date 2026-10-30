#!/usr/bin/env python3
"""Download checksum-locked official assets and build the PiPER-X family.

URDF inertias and joint limits are preserved. Mounts, controllers and cameras
are benchmark configuration, not a hardware-calibrated commercial assembly.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
LOCK = Path(__file__).with_name('piperx_assets.lock.json')
SCHEMA = 'piperx-family.v1'
GEOMETRY_RECIPE = 'convex-finger-r150-h128-t01-v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose(origin):
    if origin is None:
        return {}
    q = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_quat()
    return {'pos': origin.get('xyz', '0 0 0'), 'quat': ' '.join(map(str, q[[3, 0, 1, 2]]))}


def new_model(name):
    root = ET.Element('mujoco', model=name)
    ET.SubElement(root, 'compiler', angle='radian')
    for tag in ('asset', 'worldbody', 'actuator', 'contact'):
        ET.SubElement(root, tag)
    return root


class Builder:
    def __init__(self, destination, offline=False):
        self.destination = destination.resolve()
        self.destination.mkdir(parents=True, exist_ok=True)
        self.lock = json.loads(LOCK.read_text())
        self.offline = offline
        self.meshes = {}
        self.geometry_report = {}

    def fetch(self, key):
        item = self.lock['files'][key]
        target = self.destination / 'source' / key
        if not target.exists():
            if self.offline:
                raise FileNotFoundError(f'Offline source missing: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(item['url'], timeout=90) as response:
                        data = response.read()
                    if hashlib.sha256(data).hexdigest() != item['sha256']:
                        raise ValueError(f'Upstream checksum mismatch: {key}')
                    temp = target.with_suffix(target.suffix + '.download')
                    temp.write_bytes(data)
                    temp.replace(target)
                    print('Downloaded', key, flush=True)
                    break
                except (OSError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(1 + attempt)
        if sha(target) != item['sha256']:
            raise ValueError(f'Cached source checksum mismatch: {target}; remove it and rebuild')
        return target

    def geometry(self, mesh):
        import trimesh
        key = 'arm/' + mesh.get('filename').split('agx_arm_urdf/', 1)[1]
        path = self.fetch(key)
        scene = trimesh.load(str(path), force='scene')
        geometry = scene.to_geometry()
        geometry.apply_scale(np.fromstring(mesh.get('scale', '1 1 1'), sep=' '))
        return key, geometry

    def mesh_assets(self, link):
        """Use official STL geometry, with convex decomposition for contact.

        Render the very same convex pieces. This makes the displayed contact
        boundary identical to MuJoCo's boundary, without hidden grasp aids.
        Source-surface error is measured and bounded for the fingers.
        """
        import trimesh
        import coacd
        name = link.get('name')
        if name in self.meshes:
            return self.meshes[name]
        collision = link.find('collision')
        key, geometry = self.geometry(collision.find('geometry/mesh'))
        geometry.merge_vertices()
        finger = name in ('gripper_link1', 'gripper_link2')
        cache_key = hashlib.sha256((np.round(geometry.vertices, 9) + 0.0).tobytes() +
                                   geometry.faces.tobytes() + GEOMETRY_RECIPE.encode()).hexdigest()
        cache = self.destination / f'geometry_cache_{cache_key}.json'
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved.get('key') == cache_key and all(
                (self.destination / f).is_file() and sha(self.destination / f) == digest
                for f, digest in saved['files'].items()
            ):
                self.geometry_report[name] = {**saved['report'], 'source': key}
                self.meshes[name] = (saved['records'], pose(collision.find('origin')))
                return self.meshes[name]
        # Preserve concavity, especially at the finger's inner grasping face.
        print('Converting', name, flush=True)
        parts = []
        components = [geometry] if finger else geometry.split(only_watertight=False)
        for component in components:
            if len(component.vertices) < 4 or min(component.extents) < 1e-6:
                continue
            if component.is_convex or not finger:
                parts.append(component.convex_hull)
            else:
                coacd.set_log_level('error')
                result = coacd.run_coacd(coacd.Mesh(component.vertices, component.faces),
                    threshold=0.01, max_convex_hull=128, preprocess_mode='on', preprocess_resolution=150,
                    resolution=2000, mcts_iterations=20, mcts_nodes=10, mcts_max_depth=3,
                    merge=True, seed=0)
                for vertices, faces in result:
                    vertices = np.asarray(vertices)
                    if len(vertices) < 4 or np.linalg.matrix_rank(vertices - vertices.mean(axis=0), tol=1e-9) < 3:
                        continue  # zero-volume CAD fragments cannot collide
                    parts.append(trimesh.Trimesh(vertices, faces, process=True).convex_hull)
        if not parts:
            raise ValueError(f'No solid collision geometry: {key}')
        parts.sort(key=lambda p: tuple(np.round(p.centroid, 9)))
        records = []
        for i, part in enumerate(parts):
            filename = f'{name}_convex_{i}.obj'
            part.export(str(self.destination / filename))
            records.append((f'{name}_mesh{i}', filename))
        report = {'source': key, 'convex_parts': len(parts), 'bounds_m': geometry.bounds.tolist(),
                  'visual_collision_identical': True}
        if finger:
            # Deterministic dense bidirectional sample, including every source
            # vertex. This checks the approximation against the official STL.
            samples = np.vstack([geometry.vertices, trimesh.sample.sample_surface(geometry, 6000, seed=0)[0]])
            union = trimesh.util.concatenate(parts)
            # The CAD contains open seams, so its signed-distance inside/outside
            # classification is unreliable. Sample actual exposed surfaces using
            # first ray hits from six directions; internal convex seams cannot
            # be mistaken for the external boundary.
            exterior = []
            for axis in range(3):
                other = [i for i in range(3) if i != axis]
                grid = np.meshgrid(*[np.arange(geometry.bounds[0, i]-.001,
                    geometry.bounds[1, i]+.001, .0005) for i in other])
                for sign in (-1, 1):
                    origins = np.zeros((grid[0].size, 3))
                    origins[:, other] = np.column_stack([g.ravel() for g in grid])
                    origins[:, axis] = geometry.bounds[0 if sign == 1 else 1, axis] - sign*.01
                    directions = np.zeros_like(origins); directions[:, axis] = sign
                    hits, _, _ = union.ray.intersects_location(origins, directions, multiple_hits=False)
                    exterior.append(hits)
            outer = np.vstack(exterior)
            _, over, _ = trimesh.proximity.closest_point(geometry, outer)
            # Source points contained in any convex part are already covered.
            distances = np.stack([trimesh.proximity.signed_distance(p, samples) for p in parts])
            under = np.maximum(-distances.max(axis=0), 0)
            error = max(float(over.max()), float(under.max()))
            report['sampled_surface_error_m'] = error
            report['tolerance_m'] = .001
            report['exterior_ray_grid_m'] = .0005
            report['exterior_samples'] = len(outer)
            if error > .001:
                raise ValueError(f'{name}: collision approximation error {error:.6f} m > 1 mm')
        cache.write_text(json.dumps({'key': cache_key, 'records': records, 'report': report,
            'files': {f: sha(self.destination / f) for _, f in records}}, indent=2))
        self.geometry_report[name] = report
        self.meshes[name] = (records, pose(collision.find('origin')))
        return self.meshes[name]

    def add_link(self, root, parent, link, name, transform=None):
        body = ET.SubElement(parent, 'body', name=name, **(transform or {}))
        inertial = link.find('inertial')
        if inertial is not None:
            tensor = inertial.find('inertia')
            vals = {k: float(v) for k, v in tensor.attrib.items()}
            matrix = np.array([[vals['ixx'], vals['ixy'], vals['ixz']],
                               [vals['ixy'], vals['iyy'], vals['iyz']],
                               [vals['ixz'], vals['iyz'], vals['izz']]])
            origin = inertial.find('origin')
            r = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            matrix = r @ matrix @ r.T
            ET.SubElement(body, 'inertial', mass=inertial.find('mass').get('value'), pos=origin.get('xyz', '0 0 0'),
                fullinertia=' '.join(map(str, matrix[[0,1,2,0,0,1],[0,1,2,1,2,2]])))
        if link.find('collision') is not None:
            records, transform = self.mesh_assets(link)
            for i, (mesh_name, filename) in enumerate(records):
                if root.find(f"asset/mesh[@name='{mesh_name}']") is None:
                    ET.SubElement(root.find('asset'), 'mesh', name=mesh_name, file=filename)
                attrs = dict(type='mesh', mesh=mesh_name, **transform)
                ET.SubElement(body, 'geom', name=f'{name}_collision{i}', group='0',
                              friction='0.8 0.005 0.0001', margin='0', **attrs)
                ET.SubElement(body, 'geom', name=f'{name}_visual{i}', group='1',
                              contype='0', conaffinity='0', mass='0', rgba='0.75 0.78 0.82 1', **attrs)
        return body

    def write(self, root, filename):
        # robosuite resolves mesh paths against the XML directory on loading.
        for mesh in root.findall('asset/mesh'):
            path = Path(mesh.get('file'))
            if path.is_absolute():
                mesh.set('file', os.path.relpath(path, self.destination))
        ET.indent(root)
        ET.ElementTree(root).write(self.destination / filename, encoding='utf-8')

    def build(self):
        # Invalidate the previous bundle before writing any files.
        (self.destination / 'provenance.json').unlink(missing_ok=True)
        arm = ET.parse(self.fetch('arm/piper_x/urdf/piper_x_description.urdf')).getroot()
        gripper = ET.parse(self.fetch('arm/piper_x/urdf/piper_x_with_gripper_description.xacro')).getroot()
        self.fetch('arm/LICENSE')
        links = {x.get('name'): x for x in arm.findall('link')}
        joints = {x.get('name'): x for x in arm.findall('joint')}
        mobile = ET.parse(self.fetch('mobile/aloha_new_description/urdf/aloha_new.urdf')).getroot()
        self.fetch('mobile/aloha_new_description/package.xml')
        mobile_metadata = None
        for robot, arms in [('piperx', ['right']), ('dual_piperx', ['right','left']), ('mobile_dual_piperx', ['right','left'])]:
            root = new_model(robot)
            base = ET.SubElement(root.find('worldbody'), 'body', name='base')
            mount_parent = base
            if robot == 'mobile_dual_piperx':
                from scripts.agilex_wheeled_builder import add_wheeled_base
                mobile_metadata = add_wheeled_base(root, base, mobile,
                    lambda p: self.fetch('mobile/' + p), self.destination, pose)
                mount_parent = base.find("body[@name='mobile_body_Link']")
            for side in arms:
                position = '0 0 0' if robot == 'piperx' else f'0.23875 {-.3 if side == "right" else .3} {0.775 if robot == "mobile_dual_piperx" else 0}'
                parent = self.add_link(root, mount_parent, links['base_link'], side+'_base', {'pos':position})
                ET.SubElement(parent, 'site', name=side+'_center', size='.002', rgba='0 0 0 0')
                for i in range(1,7):
                    j = joints[f'joint{i}']; limit = j.find('limit'); previous = parent
                    parent = self.add_link(root, parent, links[f'link{i}'], side+('_hand' if i==6 else f'_link{i}'), pose(j.find('origin')))
                    name = f'{side}_joint{i}'
                    ET.SubElement(parent,'joint',name=name,type='hinge',axis=j.find('axis').get('xyz'),
                        limited='true',range=limit.get('lower')+' '+limit.get('upper'),damping='.1')
                    effort=float(limit.get('effort'))
                    ET.SubElement(root.find('actuator'),'motor',name=name+'_motor',joint=name,
                        ctrllimited='true',ctrlrange=f'{-effort} {effort}')
                    ET.SubElement(root.find('contact'),'exclude',body1=previous.get('name'),body2=parent.get('name'))
            if len(arms)==2:
                ET.SubElement(mount_parent,'camera',name='head_camera',pos='.24 0 1.24',xyaxes='0 -1 0 0 0 1',fovy='70')
            self.write(root, robot+'.xml')
        root = new_model('PiperXGripper')
        base = ET.SubElement(root.find('worldbody'),'body',name='gripper_root')
        glinks={x.get('name'):x for x in gripper.findall('link')}
        gjoints={x.get('name'):x for x in gripper.findall('joint')}
        flange=self.add_link(root,base,glinks['flange_link'],'flange',pose(gjoints['flange_joint'].find('origin')))
        housing=self.add_link(root,flange,glinks['gripper_base'],'housing',pose(gjoints['gripper_base_joint'].find('origin')))
        names={}
        for number in (1,2):
            j=gjoints[f'gripper_joint{number}'];limit=j.find('limit');i=number+6
            body=self.add_link(root,housing,glinks[f'gripper_link{number}'],f'finger{i}',pose(j.find('origin')))
            ET.SubElement(body,'joint',name=f'finger_joint{i}',type='slide',axis=j.find('axis').get('xyz'),
                range=limit.get('lower')+' '+limit.get('upper'),limited='true',damping='1')
            ET.SubElement(root.find('actuator'),'position',name=f'finger_actuator{i}',joint=f'finger_joint{i}',
                kp='300',ctrllimited='true',ctrlrange=limit.get('lower')+' '+limit.get('upper'),
                forcelimited='true',forcerange='-10 10')
            names[f'finger{i}']=[g.get('name') for g in body.findall('geom') if g.get('group')=='0']
        equality=ET.SubElement(root,'equality')
        ET.SubElement(equality,'joint',joint1='finger_joint7',joint2='finger_joint8',polycoef='0 -1 0 0 0')
        # TCP at the midpoint between the official finger origins, in housing frame.
        eef=ET.SubElement(housing,'body',name='eef',pos='0 0 .138')
        for name in ('grip_site','grip_site_cylinder','ee','ee_x','ee_y','ee_z'):
            ET.SubElement(eef,'site',name=name,size='.002',rgba='0 0 0 0',group='1')
        ET.SubElement(eef,'camera',name='eye_in_hand',pos='0 .06 -.08',xyaxes='-1 0 0 0 .919145030018 .393919298579',fovy='65')
        sensors=ET.SubElement(root,'sensor')
        ET.SubElement(sensors,'force',name='force_ee',site='ee')
        ET.SubElement(sensors,'torque',name='torque_ee',site='ee')
        self.write(root,'gripper.xml')
        xml_names = ['piperx.xml','dual_piperx.xml','mobile_dual_piperx.xml','gripper.xml']
        used = set(xml_names)
        for name in xml_names:
            used.update(m.get('file') for m in ET.parse(self.destination/name).findall('asset/mesh'))
        generated = {name: sha(self.destination/name) for name in sorted(used)}
        report={'schema':SCHEMA,'source_lock_sha256':sha(LOCK),'sources':self.lock['sources'],
            'generated_sha256':generated,'gripper_collision_names':names,'geometry':self.geometry_report,
            'mobile':mobile_metadata,'calibrated':False,
            'configuration':'Official PiPER-X arms and AGX grippers; custom fixed/mobile dual assembly on legacy Tracer/frame CAD',
            'collision_policy':'Official STL convex decomposition used identically for rendering and contact; no grasp-helper boxes',
            'mounts':{'dual_spacing_m':.6,'mobile_arm_mount_z_m':.775},
            'warning':'New embodiment invalidates old ALOHA snapshots and scores; gains, friction, TCP and mounts are simulation configuration.'}
        # Retire the obsolete entry points only after the replacement builds.
        for old_name in ('aloha.xml', 'aloha_mobile.xml'):
            (self.destination / old_name).unlink(missing_ok=True)
        for link_name in self.geometry_report:
            for path in self.destination.glob(link_name + '_convex_*.obj'):
                if path.name not in used:
                    path.unlink()
        (self.destination/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
        print('Built PiPER-X family:',self.destination,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'assets/robots/agilex')
    parser.add_argument('--offline',action='store_true',help='Require all checksum-locked sources in the local cache')
    args=parser.parse_args()
    Builder(args.output,args.offline).build()
