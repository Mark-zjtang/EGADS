import gym
import gym_carla
import torch
import os
import numpy as np
import cv2
import collections
from torchvision.utils import make_grid, save_image
from utils import write_video


def preprocess_observation_(observation, bit_depth):
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    return observation

def postprocess_observation(observation, bit_depth):
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)

def _lidar_to_observation(lidars, bit_depth):
    lidars1 = torch.transpose(lidars, 0, 2)
    lidars = torch.transpose(lidars1, 1, 2)
    preprocess_observation_(lidars, bit_depth)  # Quantise, centre and dequantise inplace
    return lidars.unsqueeze(dim=0)  # Add batch dimension

results_dir = os.path.join('video_results1')
os.makedirs(results_dir, exist_ok=True)

def set_carla_env(
        number_of_vehicles=200,
        number_of_walkers=0,
        display_size=400,
        max_past_step=1,
        dt=0.1,
        discrete=False,
        discrete_acc=[-3.0, 0.0, 3.0],
        discrete_steer=[-0.3, 0.0, 0.3],
        continuous_accel_range=[-3.0, 3.0],
        continuous_steer_range=[-0.2, 0.2],
        ego_vehicle_filter='vehicle.lincoln*',
        # ego_vehicle_filter='hero',
        port=2000,
        traffic_port=np.random.randint(2000,9000),
        town='Town03',
        task_mode='random',
        # task_mode='roundabout',
        max_time_episode=2000,
        max_waypt=12,
        obs_range=32,
        lidar_bin=0.25,
        d_behind=12,
        out_lane_thres=2.0,
        desired_speed=8,
        max_ego_spawn_times=200,
        display_route=True,
        pixor_size=64,
        pixor=False,
        predict_speed=True):
    """parameters for the gym_carla environments."""
    env_params = {
        'number_of_vehicles': number_of_vehicles,
        'number_of_walkers': number_of_walkers,
        'display_size': display_size,  # screen size of bird-eye render
        'max_past_step': max_past_step,  # the number of past steps to draw
        'dt': dt,  # time interval between two frames
        'discrete': discrete,  # whether to use discrete control space
        'discrete_acc': discrete_acc,  # discrete value of accelerations
        'discrete_steer': discrete_steer,  # discrete value of steering angles
        'continuous_accel_range': continuous_accel_range,  # continuous acceleration range
        'continuous_steer_range': continuous_steer_range,  # continuous steering angle range
        'ego_vehicle_filter': ego_vehicle_filter,  # filter for defining ego vehicle
        'port': port,  # connection port
        'traffic_port': traffic_port, # connnection traffic port
        'town': town,  # which town to simulate
        # mode of the task, [random, roundabout (only for Town03)]
        'task_mode': task_mode,
        'max_time_episode': max_time_episode,  # maximum timesteps per episode
        'max_waypt': max_waypt,  # maximum number of waypoints
        'obs_range': obs_range,  # observation range (meter)
        'lidar_bin': lidar_bin,  # bin size of lidar sensor (meter)
        'd_behind': d_behind,  # distance behind the ego vehicle (meter)
        'out_lane_thres': out_lane_thres,  # threshold for out of lane
        'desired_speed': desired_speed,  # desired speed (m/s)
        # maximum times to spawn ego vehicle
        'max_ego_spawn_times': max_ego_spawn_times,
        'display_route': display_route,  # whether to render the desired route
        'pixor_size': pixor_size,  # size of the pixor labels
        'pixor': pixor,  # whether to output PIXOR observation
        'predict_speed': predict_speed
    }
    gym_carla_env = gym.make('carla-v1',params=env_params) # default carla-v1
    return gym_carla_env



class Gym_CarlaEnv():
    def __init__(self, env_name, symbolic, seed, max_episode_length, action_repeat, bit_depth, input_names, episodes):
        self.env_name = env_name
        self.symbolic = symbolic
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth
        # Load gym_carla environment
        self.carlaEnv = set_carla_env(max_time_episode=self.max_episode_length)
        self.carlaEnv.seed(seed)
        self.input_channel = input_names
        self.episodes = episodes


    def reset(self):
        self.time = 0 # Reset internal timer
        obs = self.carlaEnv.reset()
        input_channel_to_observation = {}
        for channel in self.input_channel:
            input_channel_to_observation[channel] = _lidar_to_observation(torch.tensor(obs[channel], dtype=torch.float32), self.bit_depth)
        return input_channel_to_observation

    def step(self, action,training=False):
        reward = 0
        collision_times = 0
        vehicle_collision_distance = 0
        dt = 0.1 # time interval between two frames
        video_frames = []
        observations = {}
        for n in range(self.action_repeat):
            observation_state, reward_n, done, info = self.carlaEnv.step(action,training)
            for channel in self.input_channel:
                observations[channel] = _lidar_to_observation(torch.tensor(observation_state[channel], dtype=torch.float32), self.bit_depth)
            video_frame = torch.cat(list(observations.values()), dim=-1).cpu()
            video_frame = make_grid(video_frame + 0.5, nrow=5).numpy()
            video_frames.append(video_frame)
            reward += reward_n
            vehicle_speed = observation_state['state'][2]
            vehicle_collision_distance += vehicle_speed * dt
            if info['collision'] == 1:
                collision_times += 1
            self.time +=1
            done = done or self.time == self.max_episode_length
            if done:
               break
        # save_image(torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode{}_{}.png'.format(self.episodes, self.time)))
        observation_step = {}
        for channel in self.input_channel:
            observation_step[channel] = _lidar_to_observation(torch.tensor(observation_state[channel], dtype=torch.float32), self.bit_depth)
        observation = observation_step

        return observation, reward, done, collision_times, vehicle_collision_distance


    def render(self):
        self.carlaEnv.render()

    def close(self):
        self.carlaEnv.close()

    @property
    def observation_size(self):

        return self.carlaEnv.observation_space.shape[0] if self.symbolic else (3, 128, 128)
    @property
    def action_size(self):
        return self.carlaEnv.action_space.shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        # print("carla_sample_random {} ".format(torch.from_numpy(self.carlaEnv.action_space.sample()).shape)) # torch.size([2])
        return torch.from_numpy(self.carlaEnv.action_space.sample())

def Env(env_name, symbolic, seed, max_episode_length, action_repeat, bit_depth, input_names, episodes):
    env_name = 'carla-v0'
    return Gym_CarlaEnv(env_name, symbolic, seed, max_episode_length, action_repeat, bit_depth, input_names, episodes)

