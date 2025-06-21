"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="This script runs Real2Render2Real environments.")
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
args_cli.headless = True # Defaulted True for headless development
args_cli.enable_cameras = True # Defaulted True for rendering viewpoints
print(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_viser.franka_simulators.franka_pickup_tiger import FrankaPickupTiger
from isaaclab_viser.configs.scene_configs.franka_scene_cfg import FrankaPickupTigerCfg

import os
from pathlib import Path


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir_path, "../data")

    scene_config = FrankaPickupTigerCfg(num_envs=2, env_spacing=10.0)

    output_dir = os.path.join(data_dir, "output_data/franka_pick_tiger/")

    urdf_path = {
        'robot': Path(f'{data_dir}/franka_description/urdfs/fr3_franka_hand.urdf'),
    }
    
    FrankaPickupTiger(
        simulation_app,
        scene_config,
        urdf_path = urdf_path,
        save_data=True,
        output_dir = output_dir)
    
    
    # Using robot USD file in the scene_config and loading the same URDF separately speeds up IsaacLab init 
    # significantly (Avoids running URDF to USD converter -- just make sure the USD file and URDF correspond to one another, easiest way to guarantee this is to run USD converter on a URDF)
    
if __name__ == "__main__":
    main()
