import yaml
import os 
import numpy as np 
from argparse import Namespace
from stable_baselines3 import PPO 

from rlwrapper import ResidualRLWrapper
from pure_pursuit_controller import PurePursuitPlanner


def main():

    config_path = './config.yaml'
    
    print(f"Loading config from: {config_path}")
    with open(config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    planner = PurePursuitPlanner(conf, wb=0.3302)
    env = ResidualRLWrapper(conf, planner)

    model_path = conf.model_path

    model = PPO.load(model_path, env=env)


    # --- 4. RUN TEST LOOP ---
    num_test_episodes = 5
    
    for episode in range(num_test_episodes):
        print(f"--- Starting Episode {episode + 1} ---")
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            # Predict action using the trained model
            # deterministic=True tells the model to pick the BEST action, not explore
            action, _ = model.predict(obs, deterministic=True)
            
            # --- DEBUG: INSPECT STEERING LOGIC ---
            # We peek into the environment to see what the expert wants vs what RL wants
            try:
                raw_obs = env.env.current_obs
                if raw_obs is not None:
                    px = raw_obs['poses_x'][0]
                    py = raw_obs['poses_y'][0]
                    pt = raw_obs['poses_theta'][0]
                    
                    # Recalculate what the expert (Pure Pursuit) wants to do
                    _, base_steer = planner.plan(px, py, pt, env.base_tlad, env.base_vgain)
                    
                    # RL Action is the residual (Correction)
                    rl_steer_correction = action[0]
                    
                    print(f"Expert: {base_steer:5.2f} | RL Add: {rl_steer_correction:5.2f} | Final: {base_steer + rl_steer_correction:5.2f}")
            except Exception as e:
                pass # Ignore debug errors to keep sim running
            # -------------------------------------

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
            print("Result: TIME LIMIT / LAP LIMIT")

if __name__ == '__main__':
    main()