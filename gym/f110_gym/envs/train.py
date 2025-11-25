import yaml
import os
from argparse import Namespace
from stable_baselines3 import PPO, SAC

# Import your classes
from f110_env_extension import F110Env_Ext
from pure_pursuit_controller import PurePursuitPlanner
from utils import get_abs_path, read_config

def main():
    # Go up 2 levels to gym/maps/
    config_path = './config.yaml'
    
    print(f"Loading config from: {config_path}")
    with open(config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    training_timesteps = conf.total_timesteps

    planner = PurePursuitPlanner(conf, wb=0.3302) 

    env = F110Env_Ext(conf, planner)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

    # 5. TRAIN
    try:
        # 20k steps is a good quick test. For high performance, aim for 100k+
        model.learn(total_timesteps=training_timesteps, progress_bar=True)
    except KeyboardInterrupt:
        print("Training interrupted manually.")

    model_dir = conf.model_path
    model_path = os.path.join(model_dir, 'saved_model')

    model.save(model_path)
    print("Model saved.")
    

if __name__ == '__main__':
    main()