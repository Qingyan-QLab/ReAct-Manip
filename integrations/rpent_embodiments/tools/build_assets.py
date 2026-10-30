#!/usr/bin/env python3
"""Download pinned sources and build fixed-arm MuJoCo ports for RPent.

Fixed ALOHA retains follower mounting transforms; leaders and mobile chassis
are excluded. ARX-X5 retains the RoboDojo relative mounting transform, rotated
as a whole to the benchmark's +X-facing convention. Contact meshes are convex
decomposed for fingers; the same pieces are rendered. Servo gains are simulation
parameters, not a claim of physical system identification.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zlib

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from integrations.rpent_embodiments.models import asset_root
LOCK = ROOT/'integrations/rpent_embodiments/assets.lock.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def vec(values): return ' '.join(map(str, values))


def pose(origin):
    if origin is None: return {}
    rotation = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
    return dict(pos=origin.get('xyz', '0 0 0'), quat=vec(rotation.as_quat()[[3,0,1,2]]))


def fetch_sources(root, *, offline=False, mirror=False):
    for name, item in json.loads(LOCK.read_text())['files'].items():
        path = root/'source'/name
        if path.exists():
            if sha(path) != item['sha256']: raise ValueError(f'Source checksum mismatch: {path}')
            continue
        if offline: raise FileNotFoundError(path)
        url = item['url'].replace('https://huggingface.co/', 'https://hf-mirror.com/') if mirror else item['url']
        headers = {}
        if 'zip_entry' in item:
            start = item['offset']
            end = start + 30 + len(item['zip_entry'].encode()) + 1024 + item['compressed_size']
            headers['Range'] = f'bytes={start}-{end}'
            url += f'?part={start}'
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
            if headers and response.status != 206:
                raise ValueError('Server must support byte ranges for the robot archive')
            data = response.read()
        if headers:
            fields = struct.unpack_from('<IHHHHHIIIHH', data)
            if fields[0] != 0x04034b50: raise ValueError('Invalid ZIP local header')
            offset = 30 + fields[-2] + fields[-1]
            data = data[offset:offset+item['compressed_size']]
            if item['method'] == 8: data = zlib.decompress(data, -15)
            elif item['method'] != 0: raise ValueError('Unsupported ZIP compression')
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'Upstream checksum mismatch: {name}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print('Downloaded', name, flush=True)


class Port:
    def __init__(self, source, directory):
        self.source = source
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.xml = ET.parse(source).getroot()
        self.links = {x.get('name'): x for x in self.xml.findall('link')}
        self.joints = {x.get('name'): x for x in self.xml.findall('joint')}
        self.mesh_cache = {}

    def root(self, name):
        root = ET.Element('mujoco', model=name)
        ET.SubElement(root, 'compiler', angle='radian')
        for tag in ('asset', 'worldbody', 'actuator', 'contact'): ET.SubElement(root, tag)
        return root

    def link(self, root, parent, source_name, name, transform, *, finger=False):
        import trimesh
        link = self.links[source_name]
        body = ET.SubElement(parent, 'body', name=name, **transform)
        inertial = link.find('inertial')
        if inertial is not None:
            t = inertial.find('inertia'); origin = inertial.find('origin')
            # Rotate the full inertia tensor into the link frame.
            a = np.array([[float(t.get('ixx')),float(t.get('ixy')),float(t.get('ixz'))],
                          [float(t.get('ixy')),float(t.get('iyy')),float(t.get('iyz'))],
                          [float(t.get('ixz')),float(t.get('iyz')),float(t.get('izz'))]])
            r = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix() if origin is not None else np.eye(3)
            a = r @ a @ r.T
            ET.SubElement(body, 'inertial', mass=inertial.find('mass').get('value'),
                pos=origin.get('xyz','0 0 0') if origin is not None else '0 0 0',
                fullinertia=vec([a[0,0],a[1,1],a[2,2],a[0,1],a[0,2],a[1,2]]))
        geoms = []
        for index, collision in enumerate(link.findall('collision')):
            mesh = collision.find('geometry/mesh')
            if mesh is None: raise ValueError(f'Expected mesh collision: {source_name}')
            path = (self.source.parent/mesh.get('filename')).resolve()
            key = (str(path), mesh.get('scale', '1 1 1'), finger)
            if key not in self.mesh_cache:
                geometry = trimesh.load(str(path), force='mesh')
                geometry.apply_scale(np.fromstring(key[1],sep=' '))
                pieces = [geometry.convex_hull]
                if finger:
                    import coacd
                    coacd.set_log_level('error')
                    parts = coacd.run_coacd(coacd.Mesh(geometry.vertices, geometry.faces),
                        threshold=.01, max_convex_hull=64, resolution=2000,
                        mcts_iterations=20, mcts_nodes=10, mcts_max_depth=3, seed=0)
                    pieces = [trimesh.Trimesh(v,f).convex_hull for v,f in parts]
                files=[]
                for i, piece in enumerate(pieces):
                    filename=f'mesh_{len(self.mesh_cache)}_{i}.obj';piece.export(self.directory/filename);files.append(filename)
                self.mesh_cache[key]=files
            for i, filename in enumerate(self.mesh_cache[key]):
                mn=f'{name}_{index}_{i}'
                ET.SubElement(root.find('asset'),'mesh',name=mn,file=filename)
                attrs=dict(type='mesh',mesh=mn,**pose(collision.find('origin')))
                gn=mn+'_collision';geoms.append(gn)
                ET.SubElement(body,'geom',name=gn,group='0',friction='.8 .005 .0001',**attrs)
                ET.SubElement(body,'geom',name=mn+'_visual',group='1',contype='0',conaffinity='0',
                              rgba='.65 .68 .72 1',**attrs)
        return body, geoms

    def joint(self, root, body, source_name, name, *, gripper=False, command_range=None):
        joint=self.joints[source_name];limit=joint.find('limit')
        ET.SubElement(body,'joint',name=name,type='slide' if joint.get('type')=='prismatic' else 'hinge',
            axis=joint.find('axis').get('xyz'),range=f"{limit.get('lower')} {limit.get('upper')}",
            limited='true',damping='1',armature='.01')
        effort=float(limit.get('effort'))
        if gripper:
            ET.SubElement(root.find('actuator'),'position',name=name+'_actuator',joint=name,kp='1000',
                ctrllimited='true',ctrlrange=vec(command_range),forcelimited='true',forcerange=vec([-effort,effort]))
        else:
            ET.SubElement(root.find('actuator'),'motor',name=name+'_actuator',joint=name,
                          ctrllimited='true',ctrlrange=vec([-effort,effort]))

    def build(self, robot_id):
        aloha=robot_id=='aloha_agilex_robotwin'
        robot=self.root(robot_id); base=ET.SubElement(robot.find('worldbody'),'body',name='base')
        grip=self.root(robot_id+'_gripper');gbase=ET.SubElement(grip.find('worldbody'),'body',name='gripper')
        bias=.12 if aloha else .145;command=(-.01,.045) if aloha else (-.01,.044)
        eef=ET.SubElement(gbase,'body',name='eef',pos=f'{bias} 0 0')
        ET.SubElement(eef,'site',name='grip_site',size='.003',rgba='1 0 0 0')
        ET.SubElement(eef,'site',name='grip_site_cylinder',size='.003',rgba='1 0 0 0')
        ET.SubElement(gbase,'site',name='ft_frame',size='.003',rgba='0 0 0 0')
        sensors=ET.SubElement(grip,'sensor')
        ET.SubElement(sensors,'force',name='force_ee',site='ft_frame')
        ET.SubElement(sensors,'torque',name='torque_ee',site='ft_frame')
        for suffix in ('x','y','z'): ET.SubElement(gbase,'site',name='ee_'+suffix,size='.001',rgba='0 0 0 0')
        geoms={}
        for arm, prefix, y in [('right','fr_',-.30),('left','fl_',.30)]:
            source_prefix=prefix if aloha else ''
            if aloha: mount=pose(self.joints[prefix+'base_joint'].find('origin'))
            else: mount=dict(pos=f'0 {y} .765')
            body,_=self.link(robot,base,source_prefix+'base_link',arm+'_base',mount)
            ET.SubElement(body,'site',name=arm+'_center',size='.001',rgba='0 0 0 0')
            for i in range(1,7):
                j=self.joints[source_prefix+f'joint{i}']
                body,_=self.link(robot,body,j.find('child').get('link'),
                                 arm+'_hand' if i==6 else arm+f'_link{i}',pose(j.find('origin')))
                self.joint(robot,body,source_prefix+f'joint{i}',arm+f'_joint{i}')
            ET.SubElement(body,'camera',name='wrist_'+arm,pos='.07 .032 .065',
                          xyaxes='0 -1 0 .389418 0 .921061',fovy='65')
        # Both source grippers are symmetric in their local flange convention.
        # Keep the source joint signs; one normalized command drives two slides.
        prefix='fr_' if aloha else ''
        for i,side in [(7,'left'),(8,'right')]:
            j=self.joints[prefix+f'joint{i}']
            body,names=self.link(grip,gbase,j.find('child').get('link'),'finger'+str(i),pose(j.find('origin')),finger=True)
            self.joint(grip,body,prefix+f'joint{i}','finger_joint'+str(i),gripper=True,command_range=command)
            geoms[side]=names
        # A fixed stand replaces the excluded mobile/teleoperation structure.
        for arm in ('right','left'):
            child=base.find(f"body[@name='{arm}_base']")
            x,y,z=np.fromstring(child.get('pos'),sep=' ')
            ET.SubElement(base,'geom',name=arm+'_stand',type='box',pos=vec([x,y,z/2]),size=vec([.07,.07,z/2]),rgba='.2 .2 .2 1')
        ET.SubElement(base,'camera',name='head_camera',pos='-.15 0 1.35',xyaxes='0 -1 0 .8 0 .6',fovy='65')
        for filename,root in [('robot.xml',robot),('gripper.xml',grip)]:
            ET.indent(root);ET.ElementTree(root).write(self.directory/filename,encoding='unicode')
        data=dict(robot_id=robot_id,source_lock_sha256=sha(LOCK),generator_sha256=sha(__file__),init_qpos=[0]*12,
            gripper=dict(open_qpos=[command[1]]*2,command_range=list(command),geoms=geoms,tcp_offset=[bias,0,0]),
            conversion='fixed follower arms; source inertias and joint limits; convex collision rendering; simulation servo gains',
            files={p.name:sha(p) for p in sorted(self.directory.iterdir()) if p.suffix in ('.xml','.obj')})
        (self.directory/'manifest.json').write_text(json.dumps(data,indent=2)+'\n')
        print('Built',robot_id,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true');parser.add_argument('--mirror',action='store_true')
    args=parser.parse_args();root=asset_root();fetch_sources(root,offline=args.offline,mirror=args.mirror)
    for robot_id,path in [('aloha_agilex_robotwin','robotwin/urdf/arx5_description_isaac.urdf'),
                          ('dual_arx_x5_robodojo','robodojo/X5A.urdf')]:
        Port(root/'source'/path,root/robot_id).build(robot_id)


if __name__=='__main__': main()
