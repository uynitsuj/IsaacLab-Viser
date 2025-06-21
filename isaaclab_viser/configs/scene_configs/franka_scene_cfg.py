# import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg
from isaaclab.utils import configclass

from pathlib import Path
dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../data")

##
# Pre-defined configs
##
from isaaclab_viser.configs.scene_configs.franka_base_cfg import FrankaBaseCfg

@configclass
class FrankaPickupTigerCfg(FrankaBaseCfg):
    """Design the scene with sensors on the robot."""
    tiger = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tiger",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/assets/object_scans/tiger/tiger.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.41, 0.0, 0.1)),
    )
