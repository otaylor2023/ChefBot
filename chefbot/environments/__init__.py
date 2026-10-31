"""Environments: the worlds a setup's scene lives in (robocasa: RoboCasa kitchens; mujoco_kitchen: RoboCasa kitchens
exported to plain MuJoCo; mujoco: scene folders with their own dynamics)."""


def make_host(setup, robot, speed, out_dir, record: bool = True, camera_noise: bool = True):
    """The episode host for a setup's environment."""
    if setup.environment == "robocasa":
        from chefbot.environments.robocasa.host import RoboCasaHost

        return RoboCasaHost(setup, robot, speed, out_dir, record, camera_noise)
    if setup.environment == "mujoco_kitchen":
        from chefbot.environments.mujoco_kitchen.host import MujocoKitchenHost

        return MujocoKitchenHost(setup, robot, speed, out_dir, record, camera_noise)
    raise ValueError(f"setup {setup.name}: no host for environment {setup.environment!r}")
