import gymnasium as gym
import numpy as np
from gymnasium import spaces
from f110_gym.envs.base_classes import Integrator
from f110_gym.envs.pure_pursuit_controller import PurePursuitPlanner
from f110_gym.envs.f110_env import F110Env
from f110_gym.envs.rewards import Rewards

class ResidualRLWrapper(gym.Env):
    def __init__(self, config, planner):
        super(ResidualRLWrapper, self).__init__()
        self.conf = config
        self.planner = planner

        self.env = F110Env('f110_gym:f110-v0',
            map = self.conf.map_path,
            map_ext = self.conf.map_ext,
            num_agents = 1,
            timestep = 0.01,
            integrator = Integrator.RK4,
            seed = self.conf.seed
        )
        self.env.add_render_callback(self.planner.render_waypoints)

        # define rewards function
        self.rewards = Rewards(config, self.planner.waypoints)

        # TODO: Update action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.max_steer_residual = 0.2
        self.max_speed_residual = 2.0

        self.base_tlad = 0.8246
        self.base_vgain = 1.375

        self.num_beams = 60
        # TODO: Update observation space
        self.observation_space = spaces.Box(low = 0.0, high = 30.0, shape=(self.num_beams,), dtype=np.float32)


    def reset(self, seed=None, options=None):
        poses = np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]])
        obs, info = self.env.reset(poses = poses)
        return self.process_obs(obs), {}

    def step(self, action):
        # 1. Get Environment State
        obs_dict = self.env.current_obs
        pose_x = obs_dict['poses_x'][0]
        pose_y = obs_dict['poses_y'][0]
        pose_theta = obs_dict['poses_theta'][0]

        base_speed, base_steer = self.planner.plan(pose_x, pose_y, pose_theta, self.base_tlad, self.base_vgain)

        resid_steer = action[0] * self.max_steer_residual
        resid_speed = action[1] * self.max_speed_residual

        final_steer = base_steer + resid_steer
        final_speed = base_speed + resid_speed

        final_steer = np.clip(final_steer, -0.418, 0.418) 
        final_speed = np.clip(final_speed, 0.0, 20.0)    

        motor_commands = np.array([[final_steer, final_speed]])
        obs, reward, terminated, truncated, info = self.env.step(motor_commands)

        # TODO: Replace rewards with file calculation
        step_reward = self.rewards.get_reward(obs)

        return self.process_obs(obs), step_reward, terminated, truncated, info


    def process_obs(self, obs):
        scan = obs['scans'][0]
        indices = np.linspace(0, len(scan) - 1, self.num_beams, dtype=int)
        downsampled = scan[indices]
        return np.array(np.clip(downsampled, 0.0, 30.0), dtype=np.float32)

    def render(self, mode='human'):
        self.env.render(mode)