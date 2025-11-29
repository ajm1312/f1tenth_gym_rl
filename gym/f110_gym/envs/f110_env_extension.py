import gymnasium as gym
import numpy as np
from gymnasium import spaces
from f110_gym.envs.base_classes import Integrator
from f110_gym.envs.pure_pursuit_controller import PurePursuitPlanner, nearest_point_on_trajectory
from f110_gym.envs.f110_env import F110Env
from f110_gym.envs.rewards import Rewards


class F110Env_Ext(gym.Env):
    '''
    Extension to the F110 default environment, adding custom observation space
    and RL model extension.
    '''
    def __init__(self, config, planner):
        # Loading config and planner.
        super(F110Env_Ext, self).__init__()
        self.conf = config
        self.planner = planner

        # Setting maximum laps per episode.
        self.max_laps_per_ep = config.max_laps_per_episode

        # Loading waypoints and extracting information from file.
        self.waypoints = np.loadtxt(config.wpt_path, delimiter=config.wpt_delim, skiprows=config.wpt_rowskip)
        self.s_dist = self.waypoints[:, 1]
        self.waypoint_pts = self.waypoints[:, 3:5]
        self.right_border = self.waypoints[:, 5]
        self.left_border = self.waypoints[:, 6]
        self.ref_headings= self.waypoints[:, 7]

        # Defining car constraints.
        self.s_max = config.params.get('s_max')
        self.s_min = config.params.get('s_min')
        self.v_max = config.params.get('v_max')
        self.v_min = config.params.get('v_min')

        # Velocity  and lookahead distance
        self.alpha_v = getattr(self.conf, 'vgain', 1.2)
        self.d_la = getattr(self.conf, 'tlad', 0.6)

        # impact of residual agent on control input.
        self.alpha_rl = getattr(self.conf, 'alpha_rl', 0.6)

        # creating environment instance.
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

        # initializing action space.
        self.action_space = spaces.Box(np.array([self.s_min, self.v_min]), np.array([self.s_max, self.v_max]))

        # Creating observation space.
        self.num_beams = 60
        self.N = config.params.get('lookahead_N', 20)
        self.lookahead_step = 1
        self.obs_size = 5 + (6 * self.N)
        self.observation_space = spaces.Box(low = -1000, high = 1000, shape=(self.obs_size,))

    # Resetting position of car and environment progression.
    def reset(self, seed=None, poses=None):
        '''
        Resets environment to starting position

        Parameters
        ----------
        seed: int
            Seed for randomness
        poses: ndarray
            Custom poses for map configuration.

        Return
        ------
        obs: ndarray
            Processed observation of default environment state.
        '''
        poses = np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]])
        obs, info = self.env.reset(poses = poses)
        self.rewards.reset_state()
        return self.process_obs(obs), {}

    
    def step(self, action):
        '''
        Creates control inputs from Pure Pursuit planner and
        residual policy learner to send to environment. 
        Transforms observation from environment and sends
        to neural network.

        Parameters
        ----------
        action: ndarray(2)
            Steering angle and velocity from residual policy learner.

        Return
        ------
        observation: np.ndarray
            Processed observation space from step.
        step_reward: float
            Reward from timestep.
        terminated: bool
            Boolean indicating crash.
        truncated: bool
            Boolean indicating if maximum laps were reached.
        info: dict
            Dictionary containing environment diagnostic information.
        '''
        


        # Extract current observation.
        obs_dict = self.env.current_obs
        pose_x = obs_dict['poses_x'][0]
        pose_y = obs_dict['poses_y'][0]
        pose_theta = obs_dict['poses_theta'][0]

        # Control inputs from base pure pursuit controller.
        base_speed, base_steer = self.planner.plan(pose_x, pose_y, pose_theta, self.d_la, self.alpha_v)

        # Control inputs from residual policy learner.
        rl_steer = action[0] * self.alpha_rl
        rl_speed = action[1] * self.alpha_rl

        # Add RL and PP control inputs for final control input.
        final_steer = base_steer + rl_steer
        final_speed = base_speed + rl_speed

        # Clip to limits of steering and velocity.
        final_steer = np.clip(final_steer, self.s_min, self.s_max) 
        final_speed = np.clip(final_speed, self.v_min, self.v_max)    

        # Stepping environment with control inputs.
        motor_commands = np.array([[final_steer, final_speed]])
        obs, reward, terminated, truncated, info = self.env.step(motor_commands)

        # End current episode if lap count reaches 5.
        if(self.env.lap_counts[0] >= self.max_laps_per_ep):
            truncated = True

        step_reward = self.rewards.get_reward(obs)

        return self.process_obs(obs), step_reward, terminated, truncated, info


    def process_obs(self, obs):
        '''
        Transforms observation

        Parameters
        ----------
        obs: dict
            Observation space from step function.

        Return
        ------
        final_obs: ndarray
            Transformed observation space for residual policy learner to learn more efficiently from.
        '''
        
        # Default information of observation from original environment.
        x = obs['poses_x'][0]
        y = obs['poses_y'][0]
        theta = obs['poses_theta'][0]
        v_x = obs['linear_vels_x'][0]
        v_y = obs['linear_vels_y'][0]
        r = obs['ang_vels_z'][0]

        # Find nearest point on trajectory.
        projection, dist, segment, idx = nearest_point_on_trajectory(np.array([x, y]), self.waypoint_pts)
        
        # Calculate heading error.
        ref_heading = self.ref_headings[idx]
        delta_heading = theta - ref_heading
        delta_heading = (delta_heading + np.pi) % (2 * np.pi) - np.pi

        # Find deviation from optimal trajectory.
        vec_to_car_x = x - projection[0]
        vec_to_car_y = y - projection[1]

        # Find whether car is to the right or left of trajectory.
        cross_prod = (np.cos(ref_heading) * vec_to_car_y) - (np.sin(ref_heading) * vec_to_car_x)
        if (cross_prod < 0):
            dist = -dist

        # Put variables into vector for first half of observation.
        state_vec = np.array([dist, delta_heading, v_x, v_y, r], dtype=np.float32)

        # Find limit of waypoints for edge case.
        traj_vec = []
        max_idx = len(self.waypoints)
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Find N trajectory points in front of current position.
        for i in range(self.N):

            # Use current idx to find parameters for point s in trajectory.
            lookahead_idx = (idx + (i * self.lookahead_step)) % max_idx
            pt_ref = self.waypoint_pts[lookahead_idx]
            trk_psi = self.ref_headings[lookahead_idx]
            d_right = self.right_border[lookahead_idx]
            d_left = self.left_border[lookahead_idx]
            
            # Find normal vector to current point s on waypoint.
            nx = -np.sin(trk_psi)
            ny = np.cos(trk_psi)
            
            pt_left = pt_ref + np.array([nx * d_left, ny * d_left])
            pt_right = pt_ref - np.array([nx * d_right, ny * d_right])
            
            # Format points in terms of local car frame. 
            for p in [pt_ref, pt_left, pt_right]:
                dx = p[0] - x
                dy = p[1] - y
                local_x = dx * cos_t + dy * sin_t
                local_y = -dx * sin_t + dy * cos_t
                traj_vec.extend([local_x, local_y])

        # Appending local trajectory to obtain final observation.
        final_obs = np.concatenate([state_vec, np.array(traj_vec, dtype=np.float32)])
        
        return final_obs

    def render(self, mode='human'):
        self.env.render(mode)