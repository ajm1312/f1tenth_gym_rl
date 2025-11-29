import os
import numpy as np
from argparse import Namespace
from f110_gym.envs.pure_pursuit_controller import nearest_point_on_trajectory


class Rewards:
    '''
    Reward class for the reinforcement learning agent. 
    '''

    def __init__(self, config, raceline):
        super(Rewards, self).__init__()

        # Importing config and raceline.
        self.conf = config
        self.params = config.params
        self.raceline = raceline

        # Extracting raceline information.
        self.s_dist = raceline[:, 1]
        self.raceline_pts = raceline[:, 3:5]
        self.d_left = raceline[:, 5]
        self.d_right = raceline[:, 6]
        self.ref_headings = raceline[:, 7] 
        self.track_length = self.s_dist[-1]

        # Extracting weights and params for reward functions.
        self.alpha_dev = getattr(self.conf, 'alpha_dev', 1.0)
        self.alpha_heading = getattr(self.conf, 'alpha_heading', 0.25)
        self.tau_dev = getattr(self.conf, 'tau_dev', 0.1)
        self.tau_heading = getattr(self.conf, 'tau_heading', 0.0)
        
        # Defining maximum values for normalization.
        self.v_max = self.params.get('v_max', 20.0)
        self.t_sim = 0.01
        self.max_heading = np.pi

        # Save previous s.
        self.prev_s = None

    def get_reward(self, obs):
        '''
        Calculate reward for the current timestep.

        Parameters
        ----------
        obs: dict
            observation after action at timestep s

        Return
        ------
        r_tot: float
            total reward calculated at timestep s
        '''

        
        # Obtaining values from current observation.
        v_x = obs['linear_vels_x'][0]
        v_y = obs['linear_vels_y'][0]
        collision = obs['collisions'][0]
        x = obs['poses_x'][0]
        y = obs['poses_y'][0]
        heading = obs['poses_theta'][0]

        # Calculating closest point on trajectory
        projection, dist, segment, idx = nearest_point_on_trajectory(np.array([x, y]), self.raceline_pts)
        ref_heading = self.ref_headings[idx]
        vec_x = x - projection[0]
        vec_y = y - projection[1]
        cross_prod = (np.cos(ref_heading) * vec_y) - (np.sin(ref_heading) * vec_x)

        # Determine whether car is left or right of raceline.
        signed_dist = dist if cross_prod > 0 else -dist

        # Calling positive reward functions.
        r_adv = self.get_advance_reward(idx)
        r_speed = self.get_speed_reward(v_x, v_y)

        r_pos = r_adv + r_speed
        r_tot = r_pos

        # Calling penalty functions.
        r_dev = self.get_deviation_penalty(r_pos, signed_dist, idx)
        r_tot -= r_dev
        r_heading = self.get_heading_penalty(r_pos, heading, idx)
        r_tot -= r_heading
        r_coll = self.get_collision_penalty(collision)
        if (r_coll != 0.0):
            r_tot = -100.0

        # Returning total reward for timestep.
        return r_tot


    def get_advance_reward(self, idx):
        '''
        Calculate reward for the current timestep.

        Parameters
        ----------
        idx: int
            Index of closest point on raceline trajectory.

        Return
        ------
        r_adv: float
            Advancement reward calculated at timestep s.
        '''

        # Retrieve the cumulative track distance (s-coordinate) for the current index.
        curr_s = self.s_dist[idx]

        if (self.prev_s is None):
            # Handle the first timestep where no previous state exists.
            delta_s = 0.0
        else:
            # Calculate the distance traveled along the track centerline.
            delta_s = curr_s - self.prev_s

            # Handle case where car finished lap. Adding track_length to find correct distance traveled.
            if delta_s < -self.track_length / 2:
                delta_s += self.track_length 
            elif delta_s > self.track_length / 2:
                delta_s -= self.track_length

        # Normalize advancement reward.
        denominator = self.v_max * self.t_sim
        r_adv = delta_s / denominator

        # Save current s 
        self.prev_s = curr_s
        
        return r_adv

    def get_speed_reward(self, v_x, v_y):   
        '''
        Speed reward at timestep.

        Parameters
        ----------
        v_x: float
            Velocity in x direction.
        v_y: float
            Velocity in y direction.

        Return
        ------
        r_speed: velocity reward at timestep v

        '''
        v = np.sqrt(v_x**2 + v_y**2)
        r_speed = v / self.v_max
        return r_speed

    # Penalty for collisions against track border.
    def get_collision_penalty(self, collision):
        '''
        Collision penality.

        Parameters
        ----------
        collision: int
            flag whether collision has occurred.

        Return
        ------
        float
            0.0 if no collision, 1.0 if collision.

        '''
        if (collision):
            return 1.0
        return 0.0

    # Penalty for lateral deviation from reference line.
    def get_deviation_penalty(self, step_rew, dist, idx):
        '''
        Deviation penality.

        Parameters
        ----------
        step_rew: float
            Current positive reward.
        dist: float
            distance from raceline.
        idx: int
            current index of closest point on raceline.

        Return
        ------
        r_dev: float
            Penality for deviation from raceline.

        '''
        # Determining half width of track relative to car.
        if dist >= 0:
            half_width = self.d_left[idx]
        else:
            half_width = self.d_right[idx]

        # Changing deviation to a percentage
        deviation_pct = abs(dist) / half_width

        # Threshold value.
        if (deviation_pct < self.tau_dev):
            deviation_pct = 0.0
        
        # Calculate deviation penalty.
        r_dev = deviation_pct * step_rew * self.alpha_dev
        
        return r_dev

    def get_heading_penalty(self, step_rew, heading, idx):
        '''
        Deviation penality.

        Parameters
        ----------
        step_rew: float
            Current positive reward.
        heading: float
            Current heading angle.
        idx: int
            Current index of closest point on raceline.

        Return
        ------
        r_heading: float
            Penality from optimal heading from raceline.

        '''

        diff = heading - self.ref_headings[idx]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        heading_diff_pct = abs(diff) / self.max_heading

        if (heading_diff_pct < self.tau_heading): 
            heading_diff_pct = 0.0

        r_heading = heading_diff_pct * step_rew * self.alpha_heading
        return r_heading
        

    def reset_state(self):
        '''
        Reset state
        '''
        self.prev_s = None


    