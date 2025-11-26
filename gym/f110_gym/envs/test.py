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

    def render_callback(env_renderer):
    # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        # x = e.cars[0].vertices[::2]
        # y = e.cars[0].vertices[1::2]
        # top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = e.left + 500
        e.score_label.y = e.top - 880
        # e.left = left - 500
        # e.right = right + 500
        # e.top = top + 500
        # e.bottom = bottom - 500

        planner.render_waypoints(env_renderer)
    
    env.env.add_render_callback(render_callback)
    model_path = conf.model_path
    model_name = conf.model_name

    model_path = os.path.join(model_path, model_name)

    model = PPO.load(model_path, env=env)

    num_test_episodes = 10
    all_lap_times = []
    
    for episode in range(num_test_episodes):
        print(f"--- Starting Episode {episode + 1} ---")
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        curr_time = 0.0
        episode_lap_times = []
        curr_lap = 0
        last_lap_time = 0
        
        while not done:

            action, states = model.predict(obs, deterministic=False)

            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Render the simulation
            env.render(mode='human')

            sim_lap_count = env.env.lap_counts[0]
            sim_time = env.env.current_time

            if sim_lap_count > curr_lap:
                # A lap was just completed!
                lap_duration = sim_time - last_lap_time
                episode_lap_times.append(lap_duration)
                all_lap_times.append(lap_duration)
                
                print(f"Lap {int(sim_lap_count)} Complete: {lap_duration:.3f} seconds")
                
                last_lap_time = sim_time
                curr_lap = sim_lap_count

        if episode_lap_times:
            avg_lap_time = sum(episode_lap_times) / len(episode_lap_times)
            print(f"Average lap time for episode {episode + 1}: {avg_lap_time}")

        print(f"Episode Finished. Steps: {steps} | Total Reward: {total_reward:.2f}")
        if terminated:
            print("Result: CRASH / FINISHED")
        elif truncated:
            print("Result: LAP LIMIT")

    if (all_lap_times):
        avg_lap_time = sum(all_lap_times) / len(all_lap_times)
        best_lap_time = min(all_lap_times)
        print(f"Average lap time for episodes: {avg_lap_time}.")
        print(f"Best lap time for episodes: {best_lap_time}.")


if __name__ == '__main__':
    main()