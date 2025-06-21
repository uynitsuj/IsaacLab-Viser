
"""Rest everything follows."""
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import schemas
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from loguru import logger
from isaaclab_viser.utils.urdf_loader import load_urdf

import numpy as np
try:
    import viser
    import viser.extras
except Exception as e: # This is needed since IsaacLab installs its own older websockets version, which is incompatible with visers.
    if e.args[0] == "No module named 'websockets.asyncio'":
        import shutil
        import websockets
        try:
            shutil.rmtree(websockets.__path__[0])
            print(f"Directory '{websockets.__path__[0]}' deleted successfully. Please run the script again.")
        except OSError as e:
            print(f"Error deleting directory '{websockets.__path__[0]}': {e}")

from pathlib import Path

from isaaclab_viser.data.batch_data_logger import BatchDataLogger

import os 
from typing import Literal, Type, Mapping, Any, Optional, List, Dict, TypedDict
from collections import deque
from copy import deepcopy
from abc import ABC, abstractmethod
from isaaclab_viser.camera_manager import CameraManager

# For Domain Randomization
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from isaaclab.sim import visual_materials_cfg, visual_materials
import isaacsim.core.utils.prims as prim_utils
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

assets_root_path = '/opt/nvidia/mdl/vMaterials_2'

mdls = [
    assets_root_path + "/Metal/Aluminum_Hammered.mdl",
    assets_root_path + "/Metal/Copper_Scratched.mdl",
    assets_root_path + "/Metal/Stainless_Steel_Milled.mdl",
    assets_root_path + "/Metal/Titanium_Foil.mdl",
    assets_root_path + "/Metal/Diamond_Plate_Double_Tear.mdl",
    assets_root_path + "/Metal/Tungsten_Knurling.mdl",
    # assets_root_path + "/Carpet/Carpet_Woven.mdl",
    assets_root_path + "/Wood/OSB_Wood_Splattered.mdl",
    assets_root_path + "/Wood/OSB_Wood.mdl",
    assets_root_path + "/Wood/Wood_Bark.mdl",    
    assets_root_path + "/Wood/Wood_Cork.mdl",  
    assets_root_path + "/Wood/Laminate_Oak.mdl",
    assets_root_path + "/Wood/Wood_Tiles_Ash_Multicolor.mdl",
    assets_root_path + "/Wood/Wood_Tiles_Ash.mdl",
    assets_root_path + "/Wood/Wood_Tiles_Beech_Multicolor.mdl",
    assets_root_path + "/Wood/Wood_Tiles_Beech.mdl",
    assets_root_path + "/Wood/Wood_Tiles_Fineline_Multicolor.mdl",
    assets_root_path + "/Wood/Wood_Tiles_Fineline.mdl",
    assets_root_path + "/Wood/Wood_Tiles_Oak_Mountain_Multicolor.mdl",
    assets_root_path + "/Wood/Wood_Tiles_Oak_Mountain.mdl",
    assets_root_path + "/Wood/Wood_Tiles_Pine_Multicolor.mdl",
    assets_root_path + "/Wood/Wood_Tiles_Pine.mdl",
    assets_root_path + "/Wood/Wood_Tiles_Walnut_Multicolor.mdl",
    assets_root_path + "/Wood/Wood_Tiles_Walnut.mdl",
]

class SplatFile(TypedDict):
    """Data loaded from an antimatter15-style splat file."""

    centers: np.ndarray[np.floating]
    """(N, 3)."""
    rgbs: np.ndarray[np.floating]
    """(N, 3). Range [0, 1]."""
    opacities: np.ndarray[np.floating]
    """(N, 1). Range [0, 1]."""
    covariances: np.ndarray[np.floating]
    """(N, 3, 3)."""


class IsaacLabViser:
    """
    IsaacLab backend renderer and physics engine with Viser frontend for rapid headless development.
    """
    
    def __init__(self,
                 simulation_app,
                 scene_config: InteractiveSceneCfg,
                 urdf_path: str = None,
                 init_viser: bool = True,
                 save_data: bool = False,
                 output_dir: str = None):
        assert scene_config is not None, "Please provide a scene configuration."
        render_cfg = sim_utils.RenderCfg(antialiasing_mode='DLAA', enable_dl_denoiser=True, dlss_mode=1, enable_shadows=False) #, enable_dlssg=True)
        sim_cfg = sim_utils.SimulationCfg(device='cuda:0', render=render_cfg) #,  physx = sim_utils.PhysxCfg(max_position_iteration_count=1, max_velocity_iteration_count=1))
        self.simulation_app = simulation_app
        self.setup_simulation(sim_cfg, scene_config, output_dir)
        self.init_viser = init_viser
        self.client = None
        self.data_logger = BatchDataLogger(
            num_processes=8,
            max_queue_size=1000,
            batch_size=self.scene_config.num_envs
        ) if save_data else None
        self.urdf_path = urdf_path
        self.setup_viser(urdf_path) if init_viser else None
        
        # Random Augmentation Init
        lights_per_env = 5
        self.sphere_lights(lights_per_env * self.scene.num_envs) # Make extensible to multi-env. Random starting location currently uniformly dist. around origin
        # self.init_robot_texture_prims() # Buggy, appears to make all envs except for 0 stay in home position (but physics says otherwise)
        
    def setup_simulation(self, sim_cfg, scene_config, output_dir):
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.scene_config = scene_config
        self.scene = InteractiveScene(self.scene_config)
        self.sim.reset()

        dir_path = os.path.dirname(os.path.realpath(__file__))
        if output_dir is None:
            self.output_dir = os.path.join(dir_path, "output_data")
        else:
            self.output_dir = output_dir
        
    def setup_viser(self, urdf_path):
        if 'viewport_camera' in self.scene.sensors:
            self.use_viewport = True
            self.isaac_viewport_camera = self.scene.sensors["viewport_camera"]
        elif 'wrist_camera' in self.scene.sensors:
            self.use_viewport = False
            self.isaac_viewport_camera = self.scene.sensors["wrist_camera"]
        
        assert "robot" in self.scene.articulations, "Please provide a robot articulation with name 'robot' in the scene configuration."
        if not isinstance(self.scene_config.robot.spawn, sim_utils.UrdfFileCfg) and urdf_path is None:
            raise ValueError("Scene config file was not made with a robot URDF file (USD instead?). Please provide manually through urdf_path.")
        elif isinstance(self.scene_config.robot.spawn, sim_utils.UrdfFileCfg) and urdf_path is not None:
            raise ValueError("Scene config file was made with a robot URDF file. Overriding the provided urdf_path.")
        elif isinstance(self.scene_config.robot.spawn, sim_utils.UrdfFileCfg):
            urdf_path = {'robot':Path(self.scene_config.robot.spawn.asset_path)}
        
        self.viser_server = viser.ViserServer()
        self.urdf = {}
        for name, up in urdf_path.items():
            self.urdf[name] = load_urdf(None,up)
        
        self.render_viewport_depth = False
        self._setup_viser_scene()
        self._setup_viser_gui()
    
    def _setup_viser_scene(self):
        self.base_frame = self.viser_server.scene.add_frame("/base", show_axes=False)
        self.urdf_vis = {}
        for name, urdf in self.urdf.items():
            self.urdf_vis[name] = viser.extras.ViserUrdf(
                self.viser_server,
                urdf, 
                root_node_name="/base"
            )
        default_joint_pos_dict = {self.scene.articulations['robot'].joint_names[i]: self.scene.articulations['robot'].data.default_joint_pos[0][i].item() for i in range(len(self.scene.articulations['robot'].data.default_joint_pos[0]))} 
        self.urdf_vis['robot'].update_cfg(default_joint_pos_dict)
        
        self.camera_manager = CameraManager(self.viser_server, self.scene)
        self.use_viewport = len(self.camera_manager.frustums) == 0
        
        self.viser_ground_plane = self.viser_server.scene.add_grid(
            name="ground_plane",
            width_segments=50,
            height_segments=50,
            position=(0, 0, 0),
        )
        
    
    def _setup_viser_gui(self):
        viewport_folder = self.viser_server.gui.add_folder("Viewport")
        with viewport_folder:
            self.isaac_viewport_viser_handle = self.viser_server.gui.add_image(np.zeros((240, 320, 3)))
            if self.render_viewport_depth:
                self.isaac_viewport_viser_handle_depth = self.viser_server.gui.add_image(np.zeros((240, 320, 3)))
            self.env_selector = self.viser_server.gui.add_dropdown(
                "Environment to View",
                [str(i) for i in range(self.scene_config.num_envs)],
                initial_value='0'
            )
            self.env = int(self.env_selector.value)
            
        stats = self.viser_server.gui.add_folder("Stats")
        with stats:
            self.render_time_ms = self.viser_server.gui.add_number("Render Time (ms): ", 0, disabled=True)
            self.sim_step_time_ms = self.viser_server.gui.add_number("Simulation Step Time (ms): ", 0, disabled=True)
            self.save_time_ms = self.viser_server.gui.add_number("Save File Time (ms): ", 0, disabled=True)
            if self.data_logger is not None: 
                self.images_per_second = self.viser_server.gui.add_number("Images Saved/Second: ", 0, disabled=True)
                self.successful_envs = self.viser_server.gui.add_number("Successful Envs: ", 0, disabled=True)
            
        controls_folder = self.viser_server.gui.add_folder("Controls")
        
        # Setup camera manager GUI
        self.camera_manager.setup_gui(viewport_folder, controls_folder)
        
        @self.env_selector.on_update
        def _(_) -> None:
            self.env = int(self.env_selector.value)
        
        @self.camera_manager.add_camera_button.on_click
        def _(_) -> None:
            self.camera_manager.handle_add_camera(self.client)
            self.use_viewport = False
                  
    def render_wrapped_impl(self):
        if self.client is not None and self.use_viewport:
            if getattr(self.isaac_viewport_camera.cfg, "cams_per_env", None) is not None: # Handle batched tiled renderer for multiple cameras per environment
                repeat_n = self.scene_config.num_envs * self.isaac_viewport_camera.cfg.cams_per_env
            else:
                repeat_n = self.scene_config.num_envs
            xyz = torch.tensor(self.client.camera.position).unsqueeze(0).repeat(repeat_n, 1)
            xyz = torch.add(xyz, self.scene.env_origins.cpu().repeat_interleave(repeat_n//self.scene_config.num_envs, dim=0))
            wxyz = torch.tensor(self.client.camera.wxyz).unsqueeze(0).repeat(repeat_n, 1)
            
            self.isaac_viewport_camera.set_world_poses(xyz, wxyz, convention="ros")
            single_cam_ids = [self.isaac_viewport_camera.cfg.cams_per_env*i for i in list(range(self.scene_config.num_envs))]
            cam_out = {}
            for key in self.isaac_viewport_camera.data.output.keys():
                cam_out[key] = self.isaac_viewport_camera.data.output[key][single_cam_ids]
            self.camera_manager.buffers['camera_0'].append(cam_out)
        if not self.use_viewport and len(self.camera_manager.frustums) > 0: # Batched Rendering for MultiTiledCameraCfg
            if getattr(self.isaac_viewport_camera.cfg, "cams_per_env", None) is not None:
                repeat_n = self.scene_config.num_envs * self.isaac_viewport_camera.cfg.cams_per_env
                if len(self.camera_manager.frustums) > self.isaac_viewport_camera.cfg.cams_per_env:
                    raise ValueError(f"Using batched rendering. Not allowed to set a number of frustums exceeding config setting cams_per_env of {self.isaac_viewport_camera.cfg.cams_per_env}")
                else:
                    if hasattr(self, 'ee_pose_w_left') and hasattr(self, 'ee_pose_w_right') and self.render_wrist_cameras: # Wrist camera pose updating
                        c2w_l, c2w_r = self.wrist_cam_tf()
                        
                        if self.isaac_viewport_camera.cfg.cams_per_env > 2: # Brittle + not generalizable, hardcoded to handle 2 gripper cam + 1 ego view
                            if len(self.camera_manager.frustums) == self.isaac_viewport_camera.cfg.cams_per_env:
                                ego_cam_id = 2
                                xyz = torch.tensor(self.camera_manager.frustums[ego_cam_id].position)
                                wxyz = torch.tensor(self.camera_manager.frustums[ego_cam_id].wxyz)
                                c2w_ego = torch.cat([wxyz, xyz], dim=-1).repeat(self.scene_config.num_envs, 1).to(c2w_l.device)
                                c2w_ego[:, 4:] = torch.add(c2w_ego[:, 4:], self.scene.env_origins)
                                interleaved = torch.stack([c2w_l, c2w_r, c2w_ego], dim=1).reshape(-1, 7)
                            else:
                                interleaved = torch.stack([c2w_l, c2w_r, c2w_r], dim=1).reshape(-1, 7)
                        else:
                            interleaved = torch.stack([c2w_l, c2w_r], dim=1).reshape(-1, 7)
                        xyzs = interleaved[:, 4:] 
                        wxyzs = interleaved[:, :4] 
                        
                        self.isaac_viewport_camera.set_world_poses(xyzs, wxyzs, convention="ros")
                        indices = [i * self.isaac_viewport_camera.cfg.cams_per_env + j for i in range(self.scene_config.num_envs) for j in range(len(self.camera_manager.frustums))]
                        cam_out = {}
                        for key in self.isaac_viewport_camera.data.output.keys():
                            cam_out[key] = self.isaac_viewport_camera.data.output[key][indices]

                        for idx, frustum in enumerate(self.camera_manager.frustums):
                            frustum_data = {}
                            for key in cam_out.keys():
                                frustum_data[key] = cam_out[key][idx::len(self.camera_manager.frustums)]
                            buffer_key = frustum.name[1:]
                            if buffer_key not in self.camera_manager.buffers.keys():
                                self.camera_manager.buffers[buffer_key] = deque(maxlen=1)
                            self.camera_manager.buffers[buffer_key].append(deepcopy(frustum_data)) # TODO: Check if removing deepcopy breaks things

                    else:
                        xyzs = []
                        wxyzs = []
                        for camera_frustum in self.camera_manager.frustums:
                            xyzs.append(camera_frustum.position)
                            wxyzs.append(camera_frustum.wxyz)
                        for i in range(self.isaac_viewport_camera.cfg.cams_per_env-len(xyzs)): # Fill up with shape[0]==cams_per_env since a pose must be given every set_world_pose call for every camera
                            xyzs.append(xyzs[-1])
                            wxyzs.append(wxyzs[-1])
                        xyzs = torch.tensor(onp.array(xyzs)).repeat(self.scene_config.num_envs, 1)
                        wxyzs = torch.tensor(onp.array(wxyzs)).repeat(self.scene_config.num_envs, 1)
                        xyzs = torch.add(xyzs, self.scene.env_origins.cpu().repeat_interleave(repeat_n//self.scene_config.num_envs, dim=0))
                        self.isaac_viewport_camera.set_world_poses(xyzs, wxyzs, convention="ros")
                        indices = [i * self.isaac_viewport_camera.cfg.cams_per_env + j for i in range(self.scene_config.num_envs) for j in range(len(self.camera_manager.frustums))]
                        cam_out = {}
                        for key in self.isaac_viewport_camera.data.output.keys():
                            cam_out[key] = self.isaac_viewport_camera.data.output[key][indices]
                        
                        for idx, frustum in enumerate(self.camera_manager.frustums):
                            frustum_data = {}
                            for key in cam_out.keys():
                                frustum_data[key] = cam_out[key][idx::len(self.camera_manager.frustums)]
                            buffer_key = frustum.name[1:]
                            if buffer_key not in self.camera_manager.buffers.keys():
                                self.camera_manager.buffers[buffer_key] = deque(maxlen=1)
                            self.camera_manager.buffers[buffer_key].append(deepcopy(frustum_data)) # TODO: Check if removing deepcopy breaks things
                            if self.show_gs_in_viser:
                                frustum.image = self.camera_manager.buffers[buffer_key][0]["rgb"][self.env].clone().cpu().detach().numpy()
        if self.init_viser and self.client is not None:
            if len(self.camera_manager.buffers[self.camera_manager.render_cam]) > 0:
                self.isaac_viewport_viser_handle.image = self.camera_manager.buffers[self.camera_manager.render_cam][0]["rgb"][self.env].clone().cpu().detach().numpy()
        
        self.sim.render()
        return
    
 
            
    def sphere_lights(self, num):
        self.lights = []
        for i in range(num):
            # "CylinderLight", "DiskLight", "DistantLight", "DomeLight", "RectLight", "SphereLight"
            prim_type = "SphereLight"
            next_free_path = omni.usd.get_stage_next_free_path(self.sim.stage, f"/World/{prim_type}", False)
            light_prim = self.sim.stage.DefinePrim(next_free_path, prim_type)
            UsdGeom.Xformable(light_prim).AddTranslateOp().Set((0.0, 0.0, 0.0))
            UsdGeom.Xformable(light_prim).AddRotateXYZOp().Set((0.0, 0.0, 0.0))
            UsdGeom.Xformable(light_prim).AddScaleOp().Set((1.0, 1.0, 1.0))
            light_prim.CreateAttribute("inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool).Set(True)
            light_prim.CreateAttribute("inputs:colorTemperature", Sdf.ValueTypeNames.Float).Set(6500.0)
            light_prim.CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float).Set(0.5)
            light_prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(30000.0)
            light_prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))
            light_prim.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(0.0)
            light_prim.CreateAttribute("inputs:diffuse", Sdf.ValueTypeNames.Float).Set(1.0)
            light_prim.CreateAttribute("inputs:specular", Sdf.ValueTypeNames.Float).Set(1.0)
            self.lights.append(light_prim)

    def randomize_lighting(self):
        for light in self.lights:
            origin = self.scene.env_origins[np.random.choice(np.arange(self.scene.num_envs))].cpu().detach().numpy()
            light.GetAttribute("xformOp:translate").Set(
                (origin[0]+np.random.uniform(-4, 4), origin[1]+np.random.uniform(-4, 4), np.random.uniform(2, 4))
            )
            scale_rand = np.random.uniform(0.5, 1.5)
            light.GetAttribute("xformOp:scale").Set((scale_rand, scale_rand, scale_rand))
            light.GetAttribute("inputs:colorTemperature").Set(np.random.normal(4500, 1500))
            light.GetAttribute("inputs:intensity").Set(np.random.normal(18000, 5000))
            light.GetAttribute("inputs:color").Set(
                (np.random.uniform(0.7, 0.99), np.random.uniform(0.7, 0.99), np.random.uniform(0.7, 0.99))
            )
    
    
    @abstractmethod
    def run_simulator(self):
        raise NotImplementedError