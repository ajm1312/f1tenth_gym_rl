import time
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from argparse import Namespace
from numba import njit
from pyglet.gl import GL_POINTS

# Gym Imports
from f110_gym.envs.base_classes import Integrator
from f110_gym.envs.utils import get_abs_path
from f110_gym.envs.f110_env import F110Env

"""
Planner Helpers
"""
@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

class PurePursuitPlanner:
    '''
    Pure Pursuit Control Planner
    '''

    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]
        
    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle


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
    plt.figure("PP Trajectory", figsize=(10, 10))

    # hiding ticks.
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
    plt.title("PP Trajectories")
    if map_img is None: plt.axis('equal')

    # Plot for velocity profile over episodes.
    plt.figure("PP Velocity Profile", figsize=(12, 7))
    
    if all_laps_data:
        for i, record in enumerate(all_laps_data):
            data = record['data']
            plt.plot(data, color='red', linewidth=1.5, alpha=0.2)
        
        plt.title("PP Velocity Profile")
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

# --- 4. EVALUATION LOOP ---
def run_pure_pursuit_eval(env, planner, work_params, num_episodes=5, max_laps=10, render=True):
    '''
    Running simulation and documenting information for each lap/episode.

    Parameters
    ----------
    env: F110Env
        F1Tenth environment instance.
    planner: PurePursuitPlanner
        The pure pursuit planner instance.
    work_params: Dict[str, float]
        Dictionary containing 'tlad' (lookahead) and 'vgain' (velocity gain).
    num_episodes: int
        Number of episodes to run.
    max_laps: int
        Maximum number of laps per episode before termination.
    render: bool
        Whether to visualize the simulation.

    Returns:
    all_lap_times: List[Float]
        All lap times during episode.
    trajectories: List[List[List[float]]]
        All (x, y) pairs at every timestep during episode.
    all_laps_data: [List[dict[str, ]]]
        Velocity information for each lap during each timestep.
    '''
    
    all_lap_times = []
    trajectories = []
    all_laps_data = [] 

    tlad = work_params['tlad']
    vgain = work_params['vgain']

    for episode in range(num_episodes):
        # Resetting environment.
        obs, info = env.reset(poses=np.array([[conf.sx, conf.sy, conf.stheta]]))
        done = False

        # Saving positional and velocity information.
        x_ep = []
        y_ep = []
        current_lap_v_buffer = []
        
        steps = 0
        last_lap_time = 0.0
        curr_lap = 0 
        
        print(f"--- Episode {episode + 1} ---")

        # Loop for one episode.
        while not done:
            # Plan using Pure Pursuit.
            speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], tlad, vgain)
            
            # Step environment with prediction from planner.
            obs, step_reward, terminated, truncated, info = env.step(np.array([[steer, speed]]))
            done = (terminated or truncated)
            steps += 1

            # Update car state and save position and velocity information
            car_state = env.sim.agents[0].state 
            x_ep.append(car_state[0])
            y_ep.append(car_state[1])
            current_lap_v_buffer.append(float(car_state[3]))

            # Save lap time
            sim_lap_count = env.lap_counts[0]
            sim_time = env.current_time

            # Check if maximum laps has been reached.
            if sim_lap_count >= max_laps:
                done = True

            # End of lap detected, save values and reset current lap time.
            if sim_lap_count > curr_lap:
                lap_duration = sim_time - last_lap_time
                all_lap_times.append(lap_duration)
                
                print(f"  Ep {episode+1} Lap {int(sim_lap_count)}: {lap_duration:.3f}s")
                
                lap_record = {
                    'ep': episode + 1,
                    'lap': int(curr_lap) + 1,
                    'data': current_lap_v_buffer
                }
                all_laps_data.append(lap_record)

                current_lap_v_buffer = []
                last_lap_time = sim_time
                curr_lap = sim_lap_count

            # Optional Render
            if (render):
                env.render(mode='human')

        # Save trajectories for end of episode.
        trajectories.append([x_ep, y_ep])
        print(f"Episode {episode+1} Finished.")

    return all_lap_times, trajectories, all_laps_data

if __name__ == '__main__':
    '''
    Main function to run and evaluation for standalone pure pursuit model.
    '''
    
    # Load config.
    parent_dir = get_abs_path()
    config_path = './config.yaml' # Ensure this points to your file
    
    with open(config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    alpha_v = getattr(conf, 'vgain', 1.2)
    d_la = getattr(conf, 'tlad', 0.6)

    work_params = {'mass': 3.463, 'lf': 0.1559, 'tlad': d_la, 'vgain': alpha_v}

    # Initialize pure pursuit and environment
    planner = PurePursuitPlanner(conf, (0.17145+0.15875))
    env = F110Env(map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)

    # Get parameters from config.
    episodes = getattr(conf, 'episode_num', 5) 
    render = getattr(conf, 'render_sim', True)
    max_laps = getattr(conf, 'max_laps', 10)

    # Renderer Callback
    def render_callback(env_renderer):
        e = env_renderer
        e.score_label.x = e.left + 500
        e.score_label.y = e.top - 880
        planner.render_waypoints(env_renderer)
    env.add_render_callback(render_callback)

    # Load Map Data for Plotting
    map_data = load_map_for_plotting(conf)

    # Run Evaluation
    all_lap_times, trajectories, all_laps_data = run_pure_pursuit_eval(env, planner, work_params, episodes, max_laps, render)

    # Plot
    plot_data(all_lap_times, trajectories, all_laps_data, map_data)