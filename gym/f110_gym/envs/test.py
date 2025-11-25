import yaml
import os 
import numpy as np 
from argparse import Namespace
from stable_baselines3 import PPO, SAC

from f110_env_extension import F110Env_Ext
from pure_pursuit_controller import PurePursuitPlanner


def main():

    config_path = './config.yaml'
    
    print(f"Loading config from: {config_path}")
    with open(config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    planner = PurePursuitPlanner(conf, wb=0.3302)
    env = F110Env_Ext(conf, planner)

    model_path = conf.model_path
    model_name = conf.model_name

    model_path = os.path.join(model_path, model_name)

    model = PPO.load(model_path, env=env)

    num_test_episodes = 5
    
    for episode in range(num_test_episodes):
        print(f"--- Starting Episode {episode + 1} ---")
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:

            action, states = model.predict(obs, deterministic=False)

            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Render the simulation
            env.render(mode='human')

        print(f"Episode Finished. Steps: {steps} | Total Reward: {total_reward:.2f}")
        if terminated:
            print("Result: CRASH / FINISHED")
        elif truncated:
            print("Result: LAP LIMIT")

if __name__ == '__main__':
    main()