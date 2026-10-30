"""Task-local corrections to imported robot geometry, before compilation."""


def adapt_robot_geometry(robot):
    model = robot.robot_model
    prefix = model.naming_prefix
    root = model.worldbody.find('body')
    if model.__class__.__name__ in ('DualPiperX', 'AgileXAloha'):
        # The shoulders sit at x=.23875, beyond the original +/- .09 stand.
        for suffix in ('fixed_stand_collision', 'fixed_stand_visual'):
            geom = root.find(f"geom[@name='{prefix}{suffix}']")
            geom.set('pos', '.119375 0 .325')
            geom.set('size', '.209375 .375 .325')
