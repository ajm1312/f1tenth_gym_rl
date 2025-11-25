import os
import numpy as np
from argparse import Namespace
from f110_gym.envs.pure_pursuit_controller import nearest_point_on_trajectory


class Rewards:
    def __init__(self, config, raceline):
        super(Rewards, self).__init__()

        self.conf = config
        self.params = config.params
        self.raceline = raceline

        self.s_dist = raceline[:, 1]
        self.raceline_pts = raceline[:, 3:5]
        self.track_widths = raceline[:, 5] + raceline[:, 6]
        self.ref_headings = raceline[:, 7] 

        self.track_length = self.s_dist[-1]

        # --- 3. PARAMS & WEIGHTS ---
        self.alpha_dev = getattr(self.conf, 'alpha_dev', 1.0)
        self.alpha_heading = getattr(self.conf, 'alpha_heading', 1.0)
        self.tau_dev = getattr(self.conf, 'tau_dev', 0.1)
        self.tau_heading = getattr(self.conf, 'tau_heading', 0.1)
        
        self.v_max = self.params.get('v_max', 20.0)
        self.t_sim = 0.01
        self.max_heading = np.pi

        self.prev_s = None

    def get_reward(self, obs):
        # Obtaining values from current observation.
        v_x = obs['linear_vels_x'][0]
        v_y = obs['linear_vels_y'][0]
        collision = obs['collisions'][0]
        x = obs['poses_x'][0]
        y = obs['poses_y'][0]
        heading = obs['poses_theta'][0]

        # Calculating closest point on trajectory
        projection, dist, segment, idx = nearest_point_on_trajectory(np.array([x, y]), self.raceline_pts)

        # Calling all individual reward functions
        r_adv = self.get_advance_reward(idx)
        r_speed = self.get_speed_reward(v_x, v_y)

        r_pos = r_adv + r_speed
        r_tot = r_pos

        r_dev = self.get_deviation_penalty(r_pos, dist, idx)
        r_tot -= r_dev

        r_heading = self.get_heading_penalty(r_pos, heading, idx)
        r_tot -= r_heading

        r_coll = self.get_collision_penalty(collision)
        r_tot -= r_coll

        return r_tot

    # Reward for track advancement.
    def get_advance_reward(self, idx):

        curr_s = self.s_dist[idx]

        if (self.prev_s is None):
            delta_s = 0.0
        else:
            delta_s = curr_s - self.prev_s

            if delta_s < -self.track_length / 2:
                delta_s += self.track_length 
            elif delta_s > self.track_length / 2:
                delta_s -= self.track_length

        denominator = self.v_max * self.t_sim
        r_adv = delta_s / denominator

        self.prev_s = curr_s
        
        return r_adv 

    # Reward for higher velocities.
    def get_speed_reward(self, v_x, v_y):   
        v = np.sqrt(v_x**2 + v_y**2)
        return v / self.v_max

    # Penalty for collisions against track border.
    def get_collision_penalty(self, collision):
        if (collision):
            return 1.0
        return 0.0

    # Penalty for lateral deviation from reference line.
    def get_deviation_penalty(self, step_rew, dist, idx):
    
        curr_width = self.track_widths[idx]

        deviation_pct = abs(dist) / curr_width

        if (deviation_pct < self.tau_dev):
            deviation_pct = 0.0
        
        return deviation_pct * step_rew * self.alpha_dev

    # Penalty for deviation from optimal heading angle.
    def get_heading_penalty(self, step_rew, heading, idx):
        diff = heading - self.ref_headings[idx]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        heading_diff_pct = abs(diff) / self.max_heading

        if (heading_diff_pct < self.tau_heading): 
            heading_diff_pct = 0.0
        
        return heading_diff_pct * step_rew * self.alpha_heading
        

    def reset_state(self):
        self.prev_s = None


    