import yaml
import os 
import numpy as np 
from argparse import Namespace
from stable_baselines3 import PPO
from PIL import Image
import matplotlib.pyplot as plt

from f110_env_extension import F110Env_Ext
from pure_pursuit_controller import PurePursuitPlanner

def load_map_for_plotting(conf):
    '''
    Loads map image for plotting.

    Parameters
    -----------
    conf: Namespace
        Config file for map path.

    Returns
    -------
    map_img: np.ndarray
        Image of the map.
    resolution: Float
        Resolution of the map image.
    origin: List[float]
        Origin of map.
    '''
    map_yaml_path = conf.map_path + '.yaml'

    # Opening map information from yaml file
    with open(map_yaml_path, 'r') as f:
        map_data = yaml.safe_load(f)

    # Extracting plotting data from image file.
    resolution = map_data['resolution']
    origin = map_data['origin']
    map_dir = os.path.dirname(map_yaml_path)
    image_name = map_data.get('image', None)

    # Loading image from path
    if image_name:
        full_image_path = os.path.join(map_dir, image_name)
    else:
        full_image_path = conf.map_path + conf.map_ext
    map_img = Image.open(full_image_path).convert("L")
    map_img = np.array(map_img)
    return map_img, resolution, origin

def evaluate_model(model, env, num_episodes=10, render=True):
    '''
    Running simulation and documenting information for each lap/episode.

    Parameters
    ----------
    model: PPO
        trained model from files
    env: F110Env_Ext
        F1Tenth environment
    num_episodes: int
        number of episodes for model to run
    render: bool
        visualization for each episode

    Returns:
    all_lap_times: List[Float]
        All lap times during episode.
    trajectories: List[List[List[float]]]
        All (x, y) pairs at every timestep during episode.
    all_laps_data: [List[dict[str, ]]]
        Velocity information for each lap during each timestep.
    '''

    print(f"\n=== Starting Evaluation: {num_episodes} Episodes ===")
    
    # Storing data from all episodes
    all_lap_times = []
    trajectories = []
    all_laps_data = [] 

    # Running loop for n episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False

        # saving position and velocity for each episode.
        x_ep = []
        y_ep = []
        current_lap_v_buffer = []
        
        # saving total reward and lap times.
        total_reward = 0.0
        steps = 0
        last_lap_time = 0.0
        curr_lap = 0 
        
        # simulation loop for one episode.
        while not done:

            # Determining action and state.
            action, states = model.predict(obs, deterministic=False)

            # Taking step for determined action.
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # Rendering episode if applicable.
            if (render):
                env.render(mode='human')

            # Saving position and velocity for time step.
            car_state = env.env.sim.agents[0].state 
            x_ep.append(car_state[0])
            y_ep.append(car_state[1])
            current_lap_v_buffer.append(float(car_state[3]))

            # Saving time and lap count.
            sim_lap_count = env.env.lap_counts[0]
            sim_time = env.env.current_time

            # If lap is finished, reset visual timer and record lap time.
            if sim_lap_count > curr_lap:
                lap_duration = sim_time - last_lap_time
                all_lap_times.append(lap_duration)
                
                print(f"  Ep {episode+1} Lap {int(sim_lap_count)}: {lap_duration:.3f}s")
                
                # Saving current lap
                lap_record = {
                    'ep': episode + 1,
                    'lap': int(curr_lap) + 1,
                    'data': current_lap_v_buffer
                }
                all_laps_data.append(lap_record)
                current_lap_v_buffer = []
                last_lap_time = sim_time
                curr_lap = sim_lap_count

        # Appending trajectories taken
        trajectories.append([x_ep, y_ep])
        print(f"Episode {episode+1} Finished.")

    return all_lap_times, trajectories, all_laps_data

def plot_data(all_lap_times, trajectories, all_laps_data, map_data):
    '''
    Provides visual plots for velocity profile and trajectory. 
    Displays table showing lap times for every lap during episode.

    Parameters
    ----------
    all_lap_times: List[float]
        All lap times from previous episodes.
    trajectories: List[List[List[float]]]
        All (x,y) coordinate pairs from the episodes.
    all_laps_data: List[Dict[str, Any]]
        All velocity information from laps during all episodes.
    map_data: Optional[Tuple[np.ndarray, float, List[float]]]
        Map information and image from simulation environment.


    Returns
    -------
    None
    '''

    map_img, res, origin = map_data
    map_img = np.flipud(map_img)

    # Plotting trajectories from simulation.
    plt.figure("RL+PP Trajectory", figsize=(10, 10))

    # hiding ticks
    plt.yticks([])
    plt.xticks([])

    # Plotting image for trajectories.
    if map_img is not None:
        # Defining dimensions and plotting.
        height, width = map_img.shape
        left = origin[0]
        bottom = origin[1]
        right = left + (width * res)
        top = bottom + (height * res)
        plt.imshow(map_img, cmap='gray', vmin=0, vmax=255, 
                  extent=[left, right, bottom, top], origin='lower')

    for i, (xs, ys) in enumerate(trajectories):
        plt.plot(xs, ys, color='r', linewidth=1, alpha=0.7)
    plt.title("RLPP Trajectories")
    if map_img is None: plt.axis('equal')

    # Plot for velocity profile over episodes.
    plt.figure("RL+PP Velocity Profile", figsize=(12, 7))
    
    if all_laps_data:
        for i, record in enumerate(all_laps_data):
            data = record['data']
            plt.plot(data, color='red', linewidth=1.5, alpha=0.2)
        
        plt.title("RL+PP Velocity Profile")
        plt.xlabel("Steps into Lap")
        plt.ylabel("Velocity (m/s)")
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No Laps Recorded", ha='center')


    # Displaying lap information with averages and best lap time.
    plt.figure("Lap Time Statistics", figsize=(6, 6))
    plt.axis('off') 
    plt.title("Lap Time Statistics")

    # Filling cells if data is present.
    if all_lap_times:
        mean_time = np.mean(all_lap_times)
        table_data = []
        for i, t in enumerate(all_lap_times):
            diff = t - mean_time
            table_data.append([f"Lap {i+1}", f"{t:.3f} s", f"{diff:+.3f} s"])
        
        table_data.append(["", "", ""]) 
        table_data.append(["Average", f"{mean_time:.3f} s", "-"])
        table_data.append(["Best", f"{min(all_lap_times):.3f} s", f"{min(all_lap_times)-mean_time:+.3f} s"])

        the_table = plt.table(cellText=table_data, colLabels=["Lap", "Time", "Delta"],
                              loc='center', cellLoc='center')
        the_table.scale(1, 1.5)
    else:
        plt.text(0.5, 0.5, "No Laps Completed", ha='center')
    # Displaying plots
    plt.show()

if __name__ == '__main__':
    '''
    Loading model and environment for testing, running environment
    and displaying information.
    '''

    # Loading config file.
    config_path = './config.yaml'
    with open(config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # Creating planner and environment.
    planner = PurePursuitPlanner(conf, wb=0.3302)
    env = F110Env_Ext(conf, planner)
    
    # Parameters for simulation.
    episodes = getattr(conf, 'episode_num', 5) 
    render = getattr(conf, 'render_sim', True)
    max_laps = getattr(conf, 'max_laps', 10)

    # Defining rendering for sim.
    def render_callback(env_renderer):
        e = env_renderer
        e.score_label.x = e.left + 500
        e.score_label.y = e.top - 880
        planner.render_waypoints(env_renderer)
    env.env.add_render_callback(render_callback)
    
    # Loading model and map data.
    model_path = os.path.join(conf.model_path, conf.model_name)
    model = PPO.load(model_path, env=env)
    map_data = load_map_for_plotting(conf)

    # Beginning simulation and plotting data.
    all_lap_times, trajectories, all_laps_data = evaluate_model(model, env, episodes, render)
    plot_data(all_lap_times, trajectories, all_laps_data, map_data)