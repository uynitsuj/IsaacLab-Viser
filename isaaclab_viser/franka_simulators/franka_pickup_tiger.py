from dataclasses import dataclass
from pathlib import Path
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from collections import deque
from copy import deepcopy

from isaaclab_viser.base import IsaacLabViser
import isaaclab_viser.utils.transforms as tf
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import subtract_frame_transforms, euler_xyz_from_quat
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

# Import utility functions for quaternion operations (assuming these exist)
# from xi.utils.math import reorient_quaternion_batch, slerp_with_clip, wrap_to_pi


@dataclass
class ManipulationConfig:
    """Configuration for manipulation phases"""
    setup_phase_steps: int = 40  # End of initial setup, begin trajectory
    grasp_phase_steps: int = 110  # Close Gripper at step
    release_phase_steps: int = 160  # Open Gripper at step  
    total_steps: int = 166  # End of trajectory
    
    ee_retracts: Dict[str, float] = None
    
    def __post_init__(self):
        self.ee_retracts = {
            'start': 0.15,
            'grasp': 0.00,
            'release': 0.17
        }


class ManipulationStateMachine:
    """Handles state transitions and actions for manipulation sequence"""
    
    def __init__(self, config: ManipulationConfig):
        self.config = config
        self.gripper_closed = False
        self.ee_goal_offset = [0.0, 0.0, 0.0, 0, -1, 0, 0]
        
    def update(self, count: int):
        """Update state machine based on current count"""
        self.gripper_closed = False
        
        if count <= self.config.setup_phase_steps:
            return
            
        # Handle height transitions
        if count > self.config.setup_phase_steps:
            self.ee_goal_offset[2] = self._interpolate_height(
                count,
                self.config.setup_phase_steps + 10,
                self.config.grasp_phase_steps - 3,
                self.config.ee_retracts['start'],
                self.config.ee_retracts['grasp']
            )
            
        if count > self.config.grasp_phase_steps:
            self.gripper_closed = True
            
        if count > self.config.grasp_phase_steps + 7:
            self.ee_goal_offset[2] = self._interpolate_height(
                count,
                self.config.grasp_phase_steps + 7,
                self.config.release_phase_steps,
                self.config.ee_retracts['grasp'],
                self.config.ee_retracts['release']
            )
            
        if count > self.config.release_phase_steps:
            self.gripper_closed = False
            
    def _interpolate_height(self, count: int, start_count: int, 
                          end_count: int, start_height: float, 
                          end_height: float) -> float:
        """Smoothly interpolate end-effector height"""
        if count <= start_count:
            return start_height
        elif count >= end_count:
            return end_height
        
        t = (count - start_count) / (end_count - start_count)
        t = t * t * (3 - 2 * t)  # Smoothstep interpolation
        return start_height + t * (end_height - start_height)


class FrankaPickupTiger(IsaacLabViser):
    def __init__(self, *args, **kwargs):
        self.debug_marker_vis = True
        super().__init__(*args, **kwargs)
        
        self.state_machine = ManipulationStateMachine(ManipulationConfig())
        self.render_wrist_cameras = False
        self.run_simulator()
    
    def run_simulator(self):
        """Main simulation loop"""
        
        self.robot_entity_cfg = SceneEntityCfg(
            "robot", 
            joint_names=[".*"], 
            body_names=["panda_hand"]
        )
        self.robot_entity_cfg.resolve(self.scene)
        
        # Create IsaacLab controller - using absolute mode for stable control
        controller_cfg = DifferentialIKControllerCfg(
            command_type="pose", 
            use_relative_mode=False, 
            ik_method="dls"
        )
        self.controller = DifferentialIKController(
            controller_cfg, 
            num_envs=self.scene.num_envs, 
            device=self.sim.device
        )
        
        if self.debug_marker_vis:
            self._setup_debug_markers()
            
        count = 0
        sim_dt = self.sim.get_physics_dt()
        self.success_envs = None
        
        while self.simulation_app.is_running() and self.successful_envs.value < 1000:
            sim_start_time = time.time()
            
            self._handle_client_connection()
            self._update_rendering(sim_start_time)
            
            # Reset if needed
            if count % self.state_machine.config.total_steps == 0:
                self._handle_reset()
                count = 0

            # Update state machine
            self.state_machine.update(count)
            
            # Handle object and robot states
            self._update_object_states(count)
            if count > self.state_machine.config.setup_phase_steps:
                self._handle_manipulation(count)
                self._log_data(count)
            else:
                self._handle_setup_phase(count)
            
            self._update_sim_stats(sim_start_time, sim_dt)
            count += 1

    def _setup_viser_gui(self):
        """Setup viser GUI elements"""
        super()._setup_viser_gui()
        # Add object frame to viser
        self.rigid_objects_viser_frame = []
        
        for idx, (name, rigid_object) in enumerate(self.scene.rigid_objects.items()):
            self.rigid_objects_viser_frame.append(
                self.viser_server.scene.add_frame(
                    name, 
                    position = rigid_object.data.default_root_state[self.env][:3].cpu().detach().numpy(), 
                    wxyz = rigid_object.data.default_root_state[self.env][3:7].cpu().detach().numpy(),
                    axes_length = 0.05,
                    axes_radius = 0.003,
                    show_axes=True
                )
            )
                    
    def _setup_viser_scene(self):
        """Setup viser scene elements"""
        super()._setup_viser_scene()
        self.tf_size_handle = 0.2
        self.transform_handles = {
            'ee': self.viser_server.scene.add_frame(
                f"tf_ee_env",
                axes_length=0.5 * self.tf_size_handle,
                axes_radius=0.01 * self.tf_size_handle,
                origin_radius=0.1 * self.tf_size_handle,
                show_axes=True
            ),
        }


    def _setup_debug_markers(self):
        """Setup visualization markers for debugging"""
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/Visuals/ee_current")
        )
        self.goal_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path="/Visuals/ee_goal")
        )

    def _handle_client_connection(self):
        """Handle client connection setup"""
        if self.client is None:
            while self.client is None:
                self.client = (self.viser_server.get_clients()[0] 
                             if len(self.viser_server.get_clients()) > 0 
                             else None)
                time.sleep(0.1)

    def _update_rendering(self, sim_start_time: float):
        """Update rendering and timing"""
        render_start_time = time.time()
        
        if hasattr(self, 'robot'):
            self._update_ee_poses()
            
        self.render_wrapped_impl()
        self.render_time_ms.value = (time.time() - render_start_time) * 1e3

    def _update_ee_poses(self):
        """Update end effector poses"""
        self.ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
    
    def _handle_reset(self):
        """Handle simulation reset and environment randomization"""
        if self.success_envs is not None:
            print(f"[INFO]: Success Envs: {self.success_envs}")
            if hasattr(self, 'data_logger'):
                self.data_logger.redir_data(self.success_envs)
                
        self._reset_robot_state()
        self._reset_object_state()
        self._update_object_states(0)
        self.scene.reset()
        
        
        print("[INFO]: Resetting state...")
        self.success_envs = torch.ones((self.scene.num_envs,), device=self.scene.env_origins.device, dtype=bool)
        
    def _reset_robot_state(self):
        """Reset robot to initial state"""
        for robot in self.scene.articulations.values():
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += self.scene.env_origins
            robot.write_root_state_to_sim(root_state)
            
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

    def _reset_object_state(self):
        """Reset objects with randomization"""
        random_xyz = (torch.rand_like(self.scene.env_origins) * 2 - 1) * 0.065
        random_xyz = torch.where(random_xyz > 0.07, 0.07, random_xyz)
        random_xyz[:, 2] = 0.0
        
        for rigid_object in self.scene.rigid_objects.values():
            root_state = rigid_object.data.default_root_state.clone()
            root_state[:, :3] += self.scene.env_origins + random_xyz
            
            # Add random rotation
            random_wxyz = torch.randn_like(root_state[:, 3:7]) * 0.05
            root_state[:, 3:7] = random_wxyz
            
            rigid_object.write_root_state_to_sim(root_state)

    def _update_object_states(self, count: int):
        """Update object states - keep objects in place for tiger pickup"""
        for idx, rigid_object in enumerate(self.scene.rigid_objects.values()):
            rigid_object.update(self.sim.get_physics_dt())
            self._update_object_visualization(rigid_object, idx)

    def _update_object_visualization(self, rigid_object, idx: int):
        """Update object visualization in viser"""
        if not self.init_viser:
            return
            
        self.rigid_objects_viser_frame[idx].position = (
            rigid_object.data.root_state_w[self.env][:3].cpu().detach().numpy() - 
            self.scene.env_origins.cpu().numpy()[self.env]
        )
        self.rigid_objects_viser_frame[idx].wxyz = (
            rigid_object.data.root_state_w[self.env][3:7].cpu().detach().numpy()
        )

    def _handle_manipulation(self, count: int):
        """Handle manipulation phase of simulation"""
        # Check success conditions at appropriate time
        if count == self.state_machine.config.release_phase_steps - 6:
            self.success_envs = self._check_success_conditions()
            
        # Update robot state
        for robot in self.scene.articulations.values():
            self.robot = robot
            self._update_robot_manipulation(count)
            
        # Update visualization and simulation
        self.sim.step(render=True)
    
    def _update_robot_manipulation(self, count: int):
        """Update robot state during manipulation"""
        # Get tiger object (assume first rigid object is the tiger)
        rigid_object = list(self.scene.rigid_objects.values())[0]
        rigid_object.update(self.sim.get_physics_dt())
        
        # Calculate target poses in robot base frame
        target_poses_b = self._calculate_target_poses(count, rigid_object)
        
        # Update transform handles
        self._update_transform_handles(target_poses_b)
        
        # Get current end effector pose
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        
        # Compute current ee frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        
        # Use absolute pose commands - much simpler and more stable
        self.controller.set_command(target_poses_b, ee_pos_b, ee_quat_b)
        
        # Get jacobian for IK computation
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.robot_entity_cfg.body_ids[0] - 1, :, self.robot_entity_cfg.joint_ids]
        
        # Compute joint commands
        joint_pos_des = self.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        
        # Set gripper positions
        self._set_gripper_positions(joint_pos_des)
        
        # Apply actions at appropriate times
        if self._should_apply_actions(count):
            self.robot.set_joint_position_target(
                joint_pos_des, 
                joint_ids=self.robot_entity_cfg.joint_ids
            )
            self.robot.write_data_to_sim()
    
    def _calculate_target_poses(self, count: int, rigid_object) -> torch.Tensor:
        """Calculate target poses for robot end effector in robot base frame"""
        # Get object pose in world frame
        obj_pos_w = rigid_object.data.root_state_w[:, :3]
        obj_quat_w = rigid_object.data.root_state_w[:, 3:7]
        
        # Get robot base pose
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        
        # Transform object pose to robot base frame
        obj_pos_b, obj_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            obj_pos_w, obj_quat_w
        )
        
        # Create target pose based on object position with offset
        target_pos_b = obj_pos_b.clone()
        target_pos_b[:, 0] += self.state_machine.ee_goal_offset[0]  # x offset
        target_pos_b[:, 1] += self.state_machine.ee_goal_offset[1]  # y offset  
        target_pos_b[:, 2] += self.state_machine.ee_goal_offset[2]  # z offset (height)
        
        # Simple grasp orientation - gripper Z-axis pointing down
        # 180° rotation around X-axis to flip Z from up to down
        target_quat_b = torch.zeros_like(obj_quat_b)
        target_quat_b[:, 0] = 0.0    # w
        target_quat_b[:, 1] = 1.0    # x (180° rotation around x-axis)
        target_quat_b[:, 2] = 0.0    # y
        target_quat_b[:, 3] = 0.0    # z
        
        # Combine position and orientation
        target_poses_b = torch.cat([target_pos_b, target_quat_b], dim=1)
        
        return target_poses_b

    def _update_transform_handles(self, target_poses: torch.Tensor):
        """Update transform handles visualization"""
        self.transform_handles['ee'].position = target_poses[self.env, :3].cpu().detach().numpy()
        self.transform_handles['ee'].wxyz = target_poses[self.env, 3:7].cpu().detach().numpy()

    def _set_gripper_positions(self, joint_pos_des: torch.Tensor):
        """Set gripper joint positions based on state"""
        if self.state_machine.gripper_closed:
            joint_pos_des[:, -2:] = 0.0  # Closed
        else:
            joint_pos_des[:, -2:] = 0.04  # Open

    def _should_apply_actions(self, count: int) -> bool:
        """Determine if actions should be applied based on current count"""
        setup_complete = count > self.state_machine.config.setup_phase_steps + 7
        return setup_complete

    def _check_success_conditions(self) -> torch.Tensor:
        """Check if manipulation was successful"""
        # Get tiger object and end effector poses
        rigid_object = list(self.scene.rigid_objects.values())[0]
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        
        # Check if tiger is above height threshold and close to gripper
        root_state = rigid_object.data.root_state_w
        close_to_gripper_pos = (ee_pose_w[:, 0:3] - root_state[:, 0:3]).norm(dim=-1) < 0.2
        above_height_thresh = root_state[:, 2] > 0.05
        
        return above_height_thresh & close_to_gripper_pos

    def _handle_setup_phase(self, count: int):
        """Handle setup phase of simulation"""
        # Only perturb during very early setup to avoid persistent jitter
        if count < self.state_machine.config.setup_phase_steps - 20:
            self._random_perturb_config()
        else:
            # Hold steady position during later setup phase
            for robot in self.scene.articulations.values():
                self.robot = robot
                joint_pos_target = self.robot.data.default_joint_pos.clone()
                joint_pos_target[:, -2:] = 0.04  # Keep Gripper Open
                self.robot.set_joint_position_target(joint_pos_target)
                self.robot.write_data_to_sim()
        
        for idx, rigid_object in enumerate(self.scene.rigid_objects.values()):
            rigid_object.update(self.sim.get_physics_dt())
            self._update_object_visualization(rigid_object, idx)
            
        if count > self.state_machine.config.setup_phase_steps - 5:
            self.sim.step(render=True)
        else:
            self.sim.step(render=False)

    def _random_perturb_config(self):
        """Apply random perturbations during setup - only during early setup phase"""
        for robot in self.scene.articulations.values():
            self.robot = robot
            # Use smaller perturbations to avoid excessive jitter
            joint_pos_target = (self.robot.data.default_joint_pos + 
                              torch.randn_like(self.robot.data.joint_pos) * 0.01)
            joint_pos_target = joint_pos_target.clamp_(
                self.robot.data.soft_joint_pos_limits[..., 0],
                self.robot.data.soft_joint_pos_limits[..., 1]
            )
            joint_pos_target[:, -2:] = 0.04  # Override Gripper to Open
            self.robot.set_joint_position_target(joint_pos_target)
            self.robot.write_data_to_sim()

    def _update_sim_stats(self, sim_start_time: float, sim_dt: float):
        """Update simulation statistics and visualization"""
        if self.init_viser:
            for name in self.urdf_vis.keys():
                self.robot = self.scene.articulations[name]
                self.robot.update(sim_dt)
                
                joint_dict = {
                    self.robot.data.joint_names[i]: 
                    self.robot.data.joint_pos[self.env][i].item() 
                    for i in range(len(self.robot.data.joint_pos[0]))
                }
                self.urdf_vis[name].update_cfg(joint_dict)
                
                if self.debug_marker_vis:
                    self._update_debug_markers()
                    
            self.isaac_viewport_camera.update(0, force_recompute=True)
            
        self.sim_step_time_ms.value = (time.time() - sim_start_time) * 1e3
    
    def _log_data(self, count: int):
        """Log data during manipulation"""
        if not hasattr(self, 'data_logger'):
            return
        
        joint_names = [*self.robot.data.joint_names[:-2], (self.robot.data.joint_names[-2])]
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        
        # Compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        
        robot_data = {
            "joint_names": joint_names,
            "joint_angles": torch.cat((
                self.robot.data.joint_pos[:, :-2], 
                self.robot.data.joint_pos[:, -2].unsqueeze(-1)
            ), dim=1).clone().cpu().detach().numpy(),
            "ee_pos": torch.cat([
                ee_pos_b, ee_quat_b
            ], dim=1).cpu().detach().numpy(),
            "gripper_binary_cmd": torch.tensor(self.state_machine.gripper_closed).repeat(self.scene.num_envs).to(self.robot.device).cpu().detach().numpy()[:, None]
        }
        
        self.data_logger.save_data(
            self.camera_manager.buffers,
            robot_data, 
            count - self.state_machine.config.setup_phase_steps - 1,
            self.output_dir
        )
        
        stats = self.data_logger.get_stats()
        self.save_time_ms.value = int(stats["save_time"]*1e3)
        self.images_per_second.value = stats['images_per_second']
        self.successful_envs.value = stats['total_successful_envs']
    
    def _update_debug_markers(self):
        """Update debug visualization markers"""
        if hasattr(self, 'ee_pose_w'):
            ee_pose_w = self.ee_pose_w
            
            # Show current end effector pose
            self.ee_marker.visualize(
                ee_pose_w[:, 0:3],
                ee_pose_w[:, 3:7]
            )
            
            # Show target pose if available
            if hasattr(self, 'target_poses'):
                target_poses_tensor = torch.tensor(self.target_poses, device=self.ee_pose_w.device)
                self.goal_marker.visualize(
                    target_poses_tensor[:, 0, 0:3] + self.scene.env_origins,
                    target_poses_tensor[:, 0, 3:7]
                )
