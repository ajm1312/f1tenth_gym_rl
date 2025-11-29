import yaml
import os
from argparse import Namespace
from stable_baselines3 import PPO, SAC

# Import your classes
from f110_env_extension import F110Env_Ext
from pure_pursuit_controller import PurePursuitPlanner
from utils import get_abs_path, read_config

if __name__ == '__main__':
    '''
    Trains model for timesteps specified in config. 
    '''

    # Loading config file.
    config_path = './config.yaml'
    with open(config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    training_timesteps = conf.total_timesteps

    # Defining planner and environment.
    planner = PurePursuitPlanner(conf, wb=0.3302) 
    env = F110Env_Ext(conf, planner)

    # Creating PPO model.
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

    # Train model for specified timesteps.
    try:
        model.learn(total_timesteps=training_timesteps, progress_bar=True)
    except KeyboardInterrupt:
        print("Training interrupted manually.")

    # Saving model to specified path
    model_dir = conf.model_path
    model_path = os.path.join(model_dir, 'saved_model')

    model.save(model_path)
    print("Model saved.")