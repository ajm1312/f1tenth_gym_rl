import gymnasium as gym
import numpy as np
from gymnasium import spaces
from f110_gym.envs.base_classes import Integrator
from f110_gym.envs.pure_pursuit_controller import PurePursuitPlanner, nearest_point_on_trajectory
from f110_gym.envs.f110_env import F110Env
from f110_gym.envs.rewards import Rewards


class ResidualRLWrapper(gym.Env):
    def __init__(self, config, planner):
        super(ResidualRLWrapper, self).__init__()
        self.conf = config
        self.planner = planner

        self.waypoints = np.loadtxt(config.wpt_path, delimiter=config.wpt_delim, skiprows=config.wpt_rowskip)

        self.s_dist = self.waypoints[:, 1]
        self.waypoint_pts = self.waypoints[:, 3:5] # X, Y
        self.right_border = self.waypoints[:, 5]
        self.left_border = self.waypoints[:, 6]
        self.ref_headings= self.waypoints[:, 7]

        self.s_max = config.params.get('s_max')
        self.s_min = config.params.get('s_min')
        self.v_max = config.params.get('v_max')
        self.v_min = config.params.get('v_min')

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

        self.action_space = spaces.Box(np.array([self.s_min, self.v_min]), np.array([self.s_max, self.v_max]))

        self.base_tlad = 0.8246
        self.base_vgain = 1.375

        self.num_beams = 60
        self.N = config.params.get('lookahead_N', 20)
        self.lookahead_step = 1
        self.obs_size = 5 + (6 * self.N)
        self.observation_space = spaces.Box(low = -1000, high = 1000, shape=(self.obs_size,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        poses = np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]])
        obs, info = self.env.reset(poses = poses)
        self.rewards.reset_state()
        return self.process_obs(obs), {}

    def step(self, action):

        obs_dict = self.env.current_obs
        pose_x = obs_dict['poses_x'][0]
        pose_y = obs_dict['poses_y'][0]
        pose_theta = obs_dict['poses_theta'][0]

        base_speed, base_steer = self.planner.plan(pose_x, pose_y, pose_theta, self.base_tlad, self.base_vgain)

        final_steer = base_steer + action[0]
        final_speed = base_speed + action[1]

        final_steer = np.clip(final_steer, self.s_min, self.s_max) 
        final_speed = np.clip(final_speed, 0.0, self.v_max)    

        motor_commands = np.array([[final_steer, final_speed]])

        obs, reward, terminated, truncated, info = self.env.step(motor_commands)

        # Ending current episode if lap count reaches 5.
        if(self.env.lap_counts[0] >= 5):
            truncated = True

        step_reward = self.rewards.get_reward(obs)

        return self.process_obs(obs), step_reward, terminated, truncated, info


    def process_obs(self, obs):
        
        x = obs['poses_x'][0]
        y = obs['poses_y'][0]
        theta = obs['poses_theta'][0]
        v_x = obs['linear_vels_x'][0]
        v_y = obs['linear_vels_y'][0]
        r = obs['ang_vels_z'][0]

        projection, dist, segment, idx = nearest_point_on_trajectory(np.array([x, y]), self.waypoint_pts)
        
        # Calculate Heading Error (delta_psi)
        ref_heading = self.ref_headings[idx]
        delta_heading = theta - ref_heading
        delta_heading = (delta_heading + np.pi) % (2 * np.pi) - np.pi

        vec_to_car_x = x - projection[0]
        vec_to_car_y = y - projection[1]

        cross_prod = (np.cos(ref_heading) * vec_to_car_y) - (np.sin(ref_heading) * vec_to_car_x)
        if (cross_prod < 0):
            dist = -dist


        state_vec = np.array([dist, delta_heading, v_x, v_y, r], dtype=np.float32)

        traj_vec = []
        
        max_idx = len(self.waypoints)
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        for i in range(self.N):
            lookahead_idx = (idx + (i * self.lookahead_step)) % max_idx
            
            pt_ref = self.waypoint_pts[lookahead_idx]
            
            trk_psi = self.ref_headings[lookahead_idx]
            d_right = self.right_border[lookahead_idx]
            d_left = self.left_border[lookahead_idx]
            
            nx = -np.sin(trk_psi)
            ny = np.cos(trk_psi)
            
            pt_left = pt_ref + np.array([nx * d_left, ny * d_left])
            pt_right = pt_ref - np.array([nx * d_right, ny * d_right])
            
            for p in [pt_ref, pt_left, pt_right]:
                dx = p[0] - x
                dy = p[1] - y
                local_x = dx * cos_t + dy * sin_t
                local_y = -dx * sin_t + dy * cos_t
                traj_vec.extend([local_x, local_y])

        final_obs = np.concatenate([state_vec, np.array(traj_vec, dtype=np.float32)])
        
        return final_obs

    def render(self, mode='human_fast'):
        self.env.render(mode)