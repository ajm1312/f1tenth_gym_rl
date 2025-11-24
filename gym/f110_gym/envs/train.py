import yaml
import os
from argparse import Namespace
from stable_baselines3 import PPO

# Import your classes
from rlwrapper import ResidualRLWrapper
from pure_pursuit_controller import PurePursuitPlanner
from utils import get_abs_path, read_config

def main():
    # Go up 2 levels to gym/maps/
    config_path = './config.yaml'
    
    print(f"Loading config from: {config_path}")
    with open(config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # 2. INIT EXPERT PLANNER
    # Using the wheelbase calc from your snippet: (0.17145+0.15875)
    planner = PurePursuitPlanner(conf, wb=0.3302) 

    # 3. WRAP ENVIRONMENT
    env = ResidualRLWrapper(conf, planner)

    # 4. SETUP PPO MODEL
    # MlpPolicy = Dense Neural Network
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

    # 5. TRAIN
    print("---------------------------------------")
    print("  Starting Residual Policy Training")
    print("---------------------------------------")
    try:
        # 20k steps is a good quick test. For high performance, aim for 100k+
        model.learn(total_timesteps=10000)
    except KeyboardInterrupt:
        print("Training interrupted manually.")

    model_dir = conf.model_path
    model_path = os.path.join(model_dir, 'saved_model')

    # 6. SAVE
    model.save(model_path)
    print("Model saved.")

    # 7. VISUALIZE RESULT
    print("Starting Test Run...")
    obs, _ = env.reset()
    done = False
    while not done:
        # Model predicts action
        action, _ = model.predict(obs)
        # Env adds action to expert planner
        obs, reward, done, truncated, info = env.step(action)
        env.render()

if __name__ == '__main__':
    main()