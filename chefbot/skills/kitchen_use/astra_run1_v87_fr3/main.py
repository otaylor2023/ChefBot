"""Task-use dispatch only; every engine uses Astra for all scene/object decisions."""
from arm_only import ArmOnly
def run(robot,object='object',where='on the counter',use='hold',then='',target=''):
    robot=ArmOnly(robot)  # PORT: arm only on the real robot (arm_only.py)
    if use=='squeeze':import squeeze_policy as policy
    elif use=='press':import press_policy as policy
    elif use=='tongs':import tongs_policy as policy
    else:import hold_policy as policy
    return policy.run(robot,object=object,where=where,use=use,then=then,target=target)
