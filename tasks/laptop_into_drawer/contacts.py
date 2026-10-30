"""Explicit surface-pair materials for laptop support contacts.

MuJoCo's default equal-priority geom mixing takes the maximum sliding friction:
changing only drawer_floor cannot override rubber feet. Pair entries isolate
support tuning from the robot's existing grasp and drawer handle contacts.
"""
import xml.etree.ElementTree as ET

# Sliding coefficients are fixed modeling choices, not measured materials.
TABLE_SLIDING_FRICTION = .65
DRAWER_SLIDING_FRICTION = .30
SUPPORT_SOLREF = (.008, 1.)  # >= 2 * the task's 2 ms simulation timestep
SUPPORT_SOLIMP = (.95, .99, .001)


def install_laptop_contacts(contact, laptop):
    """Install shell + all foot pairs once, while building the MJCF model."""
    geoms = [g.get('name') for g in laptop.get_obj().iter('geom')
             if g.get('group', '0') == '0' and g.get('contype', '1') != '0']
    for surface, sliding in (('table_top', TABLE_SLIDING_FRICTION),
                             ('drawer_floor', DRAWER_SLIDING_FRICTION)):
        for geom in geoms:
            ET.SubElement(contact, 'pair', name=f'{surface}_{geom}_support',
                          geom1=surface, geom2=geom, condim='3',
                          friction=f'{sliding} {sliding} .005 .0001 .0001',
                          solref=' '.join(map(str, SUPPORT_SOLREF)),
                          solimp=' '.join(map(str, SUPPORT_SOLIMP)))
