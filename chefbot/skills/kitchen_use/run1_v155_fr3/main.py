"""Unified pick/use policy. Dispatch depends only on requested action."""
from arm_only import ArmOnly
import hold
import press
import squeeze
import tongs

def run(robot,object='object',where='on the counter',use='hold',then='',target=''):
    robot=ArmOnly(robot)  # PORT: arm only on the real robot (arm_only.py)
    if use=='tongs':
        description=target.strip()
        if description.lower().startswith('the '):description=description[4:]
        location='on the counter'
        for separator in [' in ',' on ']:
            if separator in description:
                description,suffix=description.split(separator,1)
                location=separator.strip()+' '+suffix
                break
        return tongs.run(robot,object=description or target,where=location,tool=object)
    action={'hold':hold,'press':press,'squeeze':squeeze}.get(use)
    if action is None:return {'ok':False,'reason':'unsupported requested use'}
    return action.run(robot,object=object,where=where,use=use,then=then,target=target)
