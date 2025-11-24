import os
import yaml
import numpy as np
from argparse import Namespace
from f110_gym.envs.utils import read_config
from f110_gym.envs.pure_pursuit_controller import nearest_point_on_trajectory


class Rewards:
    def __init__(self, config, raceline):
        super(Rewards, self).__init__()
        self.conf = config
        self.params = config.params

        self.ref_headings = raceline[:, 7]
        self.track_widths = raceline[:, 5] + raceline[:, 6]
        self.s_dist = racelines[:, 1]

        #define parameters from config
        self.alpha_dev = self.conf.alpha_dev
        self.alpha_heading = self.conf.alpha_heading
        self.tau_dev = self.conf.tau_dev
        self.tau_heading = self.conf.tau_heading

        self.prev_s = None
        self.max_heading = np.pi
        self.v_max = self.params.get('v_max')
        self.prev_obs = None

        # Observation Variables

    def get_reward(self, obs):
        self.prev_obs = obs
        projection, dist, segment, idx = nearest_point_on_trajectory([self.x, self.y], self.raceline)
        v_x = obs['linear_vels_x']
        v_y = obs['linear_vels_y']
        collision = obs['collisions']
        x = obs['poses_x']
        y = obs['poses_y']
        heading = obs['poses_theta']

        self.s_dist = raceline[:, 0]
        self.raceline_pts = raceline[:, 1:3]
        self.ref_headings = raceline[:, 3]
        self.t_sim = 0.01

        r_adv = self.get_advance_reward(idx)
        r_speed = self.get_speed_reward(v_x, v_y)
        r_dev = self.get_deviation_penalty()
        r_heading = self.get_heading_penalty()
        r_coll = self.get_collision_penalty()

        return 0


    def get_advance_reward(self, idx):

        curr_s = self.s_dist[idx]

        if (self.prev_s is None):
            delta_s = 0.0
        else:
            delta_s = current_s - self.prev_s

        denominator = self.v_max * self.t_sim
        r_adv = delta_s / denominator

        self.prev_s = curr_s
        
        return r_adv 

    def get_speed_reward(self, v_x, v_y):   
        v = np.linalg.norm(v_x, v_y)
        return v / self.v_max

    def get_collision_penalty(self):
        if (collision):
            return -1
        return 0.0

    def get_deviation_penalty(self):
    
        curr_s = self.s_dist[nearest_idx]

        if (self.prev_s is None):
            self.prev_s = curr_s
            delta_s = 0.0
        else:
            delta_s = current_s - self.prev_s

        denominator = self.v_max * self.t_sim

        r_dev = delta_s / denominator

        self.prev_s = curr_s

        if (r_dev > self.tau_dev): 
            return 0.0
        
        return (-alpha_dev * r_dev)

    def get_heading_penalty(self, idx):

        diff = abs(self.heading - ref_headings)

        product = diff / self.max_heading

        if (product > self.tau_heading): 
            return 0.0
        
        return (-alpha_heading * product)

# debugging main function
if __name__ == '__main__':
    config_path = './config.yaml'
    with open(config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    

    reward = Rewards(conf, [[0,1], [0,0], [1,0], [1,1]])

    print(reward.get_reward([0,1]))


    