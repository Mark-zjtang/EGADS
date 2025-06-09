#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
import gym
#import cv2
from PIL import Image
from PIL import ImageDraw
from gym import spaces
from gym.utils import seeding
import carla
from skimage.transform import resize
from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *
print('ttc')
class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, params):
        # parameters

        self.display_size = params['display_size']  # rendering screen size
        self.max_past_step = params['max_past_step']
        self.number_of_vehicles = params['number_of_vehicles']
        self.number_of_walkers = params['number_of_walkers']
        self.dt = params['dt']
        self.task_mode = params['task_mode']
        self.max_time_episode = params['max_time_episode']
        self.max_waypt = params['max_waypt']
        self.obs_range = params['obs_range']
        self.lidar_bin = params['lidar_bin']
        self.d_behind = params['d_behind']
        self.obs_size = int(self.obs_range/self.lidar_bin)
        self.out_lane_thres = params['out_lane_thres']
        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.display_route = params['display_route']
        self.traffic_port = params['traffic_port']
        self.filter_vehicle = params['ego_vehicle_filter']
        if 'pixor' in params.keys():
            self.pixor = params['pixor']
            self.pixor_size = params['pixor_size']
            self.predict_speed = params['predict_speed']
        else:
            self.pixor = False

        # Destination
        if params['task_mode'] == 'roundabout':
            self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0],
                          [-6.48, 55.47, 0], [35.96, 3.33, 0]]
        else:
            self.dests = None

        # action and observation spaces
        self.discrete = params['discrete']
        self.discrete_act = [params['discrete_acc'],
                             params['discrete_steer']]  # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc*self.n_steer)
        else:
            self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0],
                                                     params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
                                                                                                      params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer
        observation_space_dict = {
            'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'camera_depth': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'multi_fusion': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'camera_route': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
        }
        if self.pixor:
            if self.predict_speed:
                vh_regr_channel = 8
            else:
                vh_regr_channel = 6
            observation_space_dict.update({
                'roadmap': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'vh_clas': spaces.Box(low=0, high=1, shape=(self.pixor_size, self.pixor_size, 1), dtype=np.float32),
                'vh_regr': spaces.Box(low=-5, high=5, shape=(self.pixor_size, self.pixor_size, vh_regr_channel), dtype=np.float32),
                # 'pixor_state': spaces.Box(np.array([-1000, -1000, -1, -1, -5]), np.array([1000, 1000, 1, 1, 20]), dtype=np.float32)
                'pixor_state': spaces.Box(np.array([-1000, -1000, -1, -1, -5, -10, -10]), np.array([1000, 1000, 1, 1, 20, 50, 50]), dtype=np.float32)
            })
        self.observation_space = spaces.Dict(observation_space_dict)

        # Connect to carla server and get world object
        print('connecting to Carla server...')
        self.client = carla.Client('localhost', params['port'])
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(params['town'])
        print('Carla server connected!')
        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # Get spawn points
        self.vehicle_spawn_points = list(
            self.world.get_map().get_spawn_points())
        self.walker_spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)

        # # Get spawn points in roundabout
        # self.vehicle_spawn_points = list(
        #     self.world.get_map().get_spawn_points())
        # self.chocie_vehicle_spawn_points = []
        # for i, spawn_vehilce_point in enumerate(self.vehicle_spawn_points):
        #     filter_list = [228, 257, 85, 229, 8, 248, 247, 121,232, 67, 32,22,189,71, 120,162, 30,98, 163, 118, 113, 112, 211, 192, 239]
        #     if i in filter_list:
        #         continue
        #     self.chocie_vehicle_spawn_points.append(spawn_vehilce_point)
        #
        #     self.world.debug.draw_string(spawn_vehilce_point.location, str(i), life_time=1000,
        #                                  color=carla.Color(0, 0, 255))
        # self.walker_spawn_points = []
        # for i in range(self.number_of_walkers):
        #     spawn_point = carla.Transform()
        #     loc = self.world.get_random_location_from_navigation()
        #     if (loc != None):
        #         spawn_point.location = loc
        #         self.walker_spawn_points.append(spawn_point)

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(
            self.filter_vehicle, color='49,8,8')

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Lidar sensor
        self.lidar_data = None
        self.lidar_height = 2.1
        self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels', '32')
        self.lidar_bp.set_attribute('range', '5000')

        # Camera sensor
        self.camera_img = np.zeros(
            (self.obs_size, self.obs_size, 3), dtype=np.uint8)
        # self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7)) not neolix vehicle
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.5))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set im age resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        #Camera depth sensor
        self.camera_depth_img = np.zeros(
            (self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.camera_depth_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_depth_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_depth_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_depth_bp.set_attribute('fov', '110')
        self.camera_depth_bp.set_attribute('sensor_tick', '0.02')

        #Camera semantic sensor
        self.semantic_img = np.zeros(
            (self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.semantic_img_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.semantic_img_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.semantic_img_bp.set_attribute('image_size_x', str(self.obs_size))
        self.semantic_img_bp.set_attribute('image_size_y', str(self.obs_size))
        self.semantic_img_bp.set_attribute('fov', '110')
        self.semantic_img_bp.set_attribute('sensor_tick', '0.02')



        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # If the server is set to synchronous mode, the TM must be set to synchronous mode too (for carla 0.9.10)
        self.tm = self.client.get_trafficmanager(self.traffic_port)
        print('connecting to trafficmanager_port',self.traffic_port)
        self.tm.get_port()
        self.tm.set_synchronous_mode(True)



        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0
        self.total_reward = 0

        self.collision = 0
        self.other_actor = None

        self.out_lane_dis = 0
        self.set_first_random_state = False
        self.get_init_random_state = None

        self.blocked_time = 0


        # Initialize the renderer
        self._init_renderer()

        # Get pixel grid points
        if self.pixor:
            x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(
                self.pixor_size))  # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            self.pixel_grid = np.vstack((x, y)).T

    # convert rgb (225,225,3 ) to gray (225,225) image
    def rgb_to_gray(self,rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # channel turn R G B



    def reset(self):

        # Clear sensor objects
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None
        self.camera_depth_sensor = None
        self.semantic_img_sensor = None

        # Disable sync mode
        self._set_synchronous_mode(synchronous=False)

        # Delete sensors, vehicles and walkers
        self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast',
                                'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker', 'walker.*', 'sensor.camera.depth', 'sensor.camera.semantic_segmentation'])

        # # Spawn roundabout surrounding vehicles
        # tmp_vehicle_spawn_points = self.vehicle_spawn_points.copy()
        # random.shuffle(tmp_vehicle_spawn_points)
        # random.shuffle(self.chocie_vehicle_spawn_points)
        # count = self.number_of_vehicles
        # if count > 0:
        #     for spawn_point in self.chocie_vehicle_spawn_points:
        #
        #         if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
        #             count -= 1
        #         if count <= 0:
        #             break
        # while count > 0:
        #     if self._try_spawn_random_vehicle_at(random.choice(self.chocie_vehicle_spawn_points), number_of_wheels=[4]):
        #         count -= 1

        # Spawn surrounding vehicles
        tmp_vehicle_spawn_points = self.vehicle_spawn_points.copy()
        random.shuffle(tmp_vehicle_spawn_points)
        count = self.number_of_vehicles
        if count > 0:
            for spawn_point in tmp_vehicle_spawn_points:
                if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
                count -= 1


        # Spawn pedestrians
        random.shuffle(self.walker_spawn_points)
        count = self.number_of_walkers
        if count > 0:
            for spawn_point in self.walker_spawn_points:
                if self._try_spawn_random_walker_at(spawn_point):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
                count -= 1
        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        # self.clock.tick(15)

        # # Spawn the ego vehicle in roundabout
        # ego_spawn_times = 0
        # # filter_list = [151, 129, 233, 235, 181, 61, 179, 65, 177, 183, 173, 71, 67, 223, 219, 217, 215, 213, 211, 107, 35, 89, 37, 39, 29, 41, 75, 27, 135, 25, 45, 79, 23, 17, 161, 19, 159, 157, 155, 153, 245, 137, 247, 249, 49, 251, 131, 133, 127, 85, 15, 147, 193, 143, 239, 191, 145, 189, 162, 62, 64, 226, 178, 218, 176, 222, 68, 86, 220, 70, 216, 34, 36, 78, 38, 150, 154, 156, 158, 160, 166, 164, 22, 84, 44, 24, 240, 242, 12, 80, 0, 252, 130, 250, 48, 248, 198, 196, 148, 194]
        #
        # filter_list = [228, 257, 85, 229, 8, 248, 247, 121,232, 67, 32,22,189,71, 120,162, 30,98, 163, 118, 113, 112, 211, 192, 239]
        # random.shuffle(filter_list)
        # while True:
        #     if ego_spawn_times > self.max_ego_spawn_times:
        #         self.reset()
        #
        #     if self.task_mode == 'random':
        #         transform_idx = random.choice(filter_list)
        #         transform = self.vehicle_spawn_points[transform_idx]
        #     elif self.task_mode == 'roundabout':
        #         self.start = [
        #             52.1+np.random.uniform(-5, 5), -4.2, 178.66]  # random
        #         transform = set_carla_transform(self.start)
        #     if self._try_spawn_ego_vehicle_at(transform):
        #         break
        #     else:
        #         ego_spawn_times += 1
        #         time.sleep(0.1)

        # Spawn the ego vehicle
        ego_spawn_times = 0
        # filter_list = [151, 129, 233, 235, 181, 61, 179, 65, 177, 183, 173, 71, 67, 223, 219, 217, 215, 213, 211, 107, 35, 89, 37, 39, 29, 41, 75, 27, 135, 25, 45, 79, 23, 17, 161, 19, 159, 157, 155, 153, 245, 137, 247, 249, 49, 251, 131, 133, 127, 85, 15, 147, 193, 143, 239, 191, 145, 189, 162, 62, 64, 226, 178, 218, 176, 222, 68, 86, 220, 70, 216, 34, 36, 78, 38, 150, 154, 156, 158, 160, 166, 164, 22, 84, 44, 24, 240, 242, 12, 80, 0, 252, 130, 250, 48, 248, 198, 196, 148, 194]
        filter_list = []
        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                self.reset()

            if self.task_mode == 'random':
                transform_idx = random.randint(0, len(self.vehicle_spawn_points) - 1)
                count = 0
                while transform_idx in filter_list:
                    count += 1
                    transform_idx = random.randint(0, int(len(self.vehicle_spawn_points) / 3))
                    if count > 100:
                        break
                transform = self.vehicle_spawn_points[transform_idx]

            elif self.task_mode == 'roundabout':
                self.start = [
                    52.1 + np.random.uniform(-5, 5), -4.2, 178.66]  # random
                # self.start=[52.1,-4.2, 178.66] # static
                transform = set_carla_transform(self.start)
            if self._try_spawn_ego_vehicle_at(transform):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            self.other_actor = event.other_actor
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)
        self.collision_hist = []

        # Add lidar sensor
        self.lidar_sensor = self.world.spawn_actor(
            self.lidar_bp, self.lidar_trans, attach_to=self.ego)
        self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        def get_lidar_data(data):
            self.lidar_data = data

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(
            self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

        # Add camera depth sensor
        self.camera_depth_sensor = self.world.spawn_actor(
            self.camera_depth_bp, self.camera_depth_trans, attach_to=self.ego)
        self.camera_depth_sensor.listen(lambda data: get_camera_depth_img(data))

        def get_camera_depth_img(data):
            # data.convert(carla.ColorConverter.Depth)
            data.convert(carla.ColorConverter.LogarithmicDepth)
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_depth_img = array


        # Update timesteps
        self.time_step = 0
        self.reset_step += 1
        self.total_reward = 0

        self.collision = 0
        self.other_actor = None

        self.out_lane_dis = 0

        self.blocked_time = 0


        # Enable sync mode
        self._set_synchronous_mode(synchronous=True)

        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.waypoints, _, self.vehicle_front, _, _ = self.routeplanner.run_step()

        # Set ego information for render
        self.birdeye_render.set_hero(self.ego, self.ego.id)

        return self._get_obs()

    def step(self, action, training=False):
        # Calculate acceleration and steering

        if self.discrete:
            acc = self.discrete_act[0][action//self.n_steer]
            steer = self.discrete_act[1][action % self.n_steer]
        else:
            acc = action[0]
            steer = action[1]

        # Convert acceleration to throttle and brake
        if training:
            if acc > 0:
                throttle = np.clip(acc, 0, 1)
                brake = 0
            else:
                throttle = 0
                brake = np.clip(-acc, 0, 1)
        else:
            if acc > 0:
                throttle = np.clip(acc/2, 0, 1)
                brake = 0
            else:
                throttle = 0
                brake = np.clip(-acc/0.5, 0, 1)

        action_norm = [throttle, steer]


        # Apply control
        act = carla.VehicleControl(throttle=float(
            throttle), steer=float(-steer), brake=float(brake))
        self.ego.apply_control(act)
        self.world.tick()

        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        # route planner
        self.waypoints, _, self.vehicle_front, front_vehicle_time_to_collision, around_vehicle_time_to_collision = self.routeplanner.run_step()


        terminal = self._terminal()

        # state information
        info = {
            'waypoints': self.waypoints,
            'vehicle_front': self.vehicle_front,
            'collision': self.collision
        }

        # Update timesteps
        self.time_step += 1
        self.total_step += 1
        # print(self.total_step)
        obs = self._get_obs()
        lateral_distance = obs['state'][0]
  
        return (self._get_obs(), self._get_reward(front_vehicle_time_to_collision, around_vehicle_time_to_collision, action_norm, lateral_distance), terminal, copy.deepcopy(info))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        # print('seed',seed)
        return [seed]

    def render(self):
        # lidar_surface = rgb_to_display_surface(lidar, self.display_size)
        # self.display.blit(lidar_surface, (self.display_size, 0))
        pass


    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

        Args:
          actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
          bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library()
        blueprints = blueprints.filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + \
                [x for x in blueprints if int(
                    x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(
                    bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _init_renderer(self):
        """Initialize the birdeye view renderer.
        """

        pygame.init()
        self.display = pygame.display.set_mode(
            (self.display_size*3, self.display_size),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        pixels_per_meter = self.display_size / self.obs_range
        self.pixels_per_meter = pixels_per_meter
        pixels_ahead_vehicle = (
            self.obs_range/2 - self.d_behind) * pixels_per_meter
        birdeye_params = {
            'screen_size': [self.display_size, self.display_size],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle
        }
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

        # self.clock.tick(25)
        # pygame.display.update()




    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint(
            'vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        # Add self.tm_port for multi - simulation
        if vehicle is not None:
            vehicle.set_autopilot(True, self.tm.get_port())
            return True
        return False

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(
            self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(
                walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(
                self.world.get_random_location_from_navigation())
            # random max speed
            # max speed between 1 and 2 (default is 1.4 m/s)
            walker_controller_actor.set_max_speed(1 + random.random())
            return True
        return False

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break
        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
          filt: the filter indicating what type of actors we'll look at.

        Returns:
          actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw/180*np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array(
                [[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)],
                          [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + \
                np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict



    def perspective_change(self, image, obs_size):
        # perspectivate_matrix
        matrix = [[ 6.25000000e-02, -4.68750000e-01,  6.00000000e+01],
                 [ 0.00000000e+00, -4.84375000e-01,  7.00000000e+01],
                 [-0.00000000e+00, -7.32421875e-03,  1.00000000e+00]]

        matrix = np.array(matrix)
        img_output = cv2.warpPerspective(image, matrix, (obs_size, obs_size))
        return img_output

    def _get_obs(self):
        """Get the observations."""

        # Birdeye rendering
        self.birdeye_render.vehicle_polygons = self.vehicle_polygons
        self.birdeye_render.walker_polygons = self.walker_polygons
        self.birdeye_render.waypoints = self.waypoints
        # birdeye view with roadmap and actors
        birdeye_render_types = ['roadmap', 'actors']
        if self.display_route:
            birdeye_render_types.append('waypoints')
        self.birdeye_render.render(self.display, birdeye_render_types)
        birdeye = pygame.surfarray.array3d(self.display)
        birdeye = birdeye[0: self.display_size,:, :]
        birdeye = display_to_rgb(birdeye, self.obs_size)


        # Roadmap
        if self.pixor:
            roadmap_render_types = ['roadmap']
            if self.display_route:
                roadmap_render_types.append('waypoints')
            self.birdeye_render.render(self.display, roadmap_render_types)
            roadmap = pygame.surfarray.array3d(self.display)
            roadmap = roadmap[0:self.display_size, :, :]
            roadmap = display_to_rgb(roadmap, self.obs_size)
            # Add ego vehicle
            for i in range(self.obs_size):
                for j in range(self.obs_size):
                    if abs(birdeye[i, j, 0] - 255) < 20 and abs(birdeye[i, j, 1] - 0) < 20 and abs(birdeye[i, j, 0] - 255) < 20:
                        roadmap[i, j, :] = birdeye[i, j, :]

        # Display birdeye image
        birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
        self.display.blit(birdeye_surface, (0, 0))

        # Display camera image
        camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
        camera_surface = rgb_to_display_surface(camera, self.display_size)
        self.display.blit(camera_surface, (self.display_size*2, 0))

        # Lidar image generation
        point_cloud = []
        # Get point cloud data
        for location in self.lidar_data:
            # print(location)
            point_cloud.append([location.point.x, location.point.y, location.point.z])
            # self.clock.tick(15)
        point_cloud = np.array(point_cloud)
        # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
        # and z is set to be two bins.
        x_bins = np.arange(-self.d_behind,
                           self.obs_range - self.d_behind + self.lidar_bin,
                           self.lidar_bin)
        y_bins = np.arange(-self.obs_range / 2,
                           self.obs_range / 2 + self.lidar_bin,
                           self.lidar_bin)
        z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]
        # Get lidar image according to the bins
        lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
        lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
        lidar_zeros = np.zeros((self.obs_size, self.obs_size, 1))

        # Add the waypoints to lidar image

        if self.display_route:
            wayptimg = (birdeye[:, :, 0] <= 10) * \
                (birdeye[:, :, 1] <= 10) * (birdeye[:, :, 2] >= 240)
        else:
            wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix

        wayptimg = np.expand_dims(wayptimg, axis=2)
        wayptimg = np.flip(np.rot90(wayptimg, 2))

        lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
        lidar_second = np.expand_dims(lidar[:, :, 1], axis=2)
        lidar_second = np.concatenate((lidar_zeros, lidar_second), axis=-1)
        lidar_second = np.flip(lidar_second, axis=0)
        lidar_second = np.concatenate((lidar_second, wayptimg), axis=-1)
        lidar_second = lidar_second * 255
        lidar_noground = lidar_second
        lidar = np.flip(lidar, axis=0)


        # Get the final lidar image
        lidar = np.concatenate((lidar, wayptimg), axis=2)
        lidar = lidar * 255
        self.lidar = lidar
        # Display lidar image
        lidar_noground_surface = rgb_to_display_surface(lidar_noground, self.display_size)
        self.display.blit(lidar_noground_surface, (self.display_size, 0))





        # Display camera_route
        # self.birdeye_render.render(self.display, birdeye_render_types)
        # camera_route = pygame.surfarray.array3d(self.display)
        # camera_route = camera_route[self.display_size*2:self.display_size*3, :, :]
        # camera_route = display_to_rgb(camera_route, self.obs_size)
        # camera_routing = self.perspective_change(camera_route, self.obs_size)
        # camera_route = resize(camera_routing, (self.obs_size, self.obs_size))
        # camera_route_waypoint = (camera_route[:, :, 0] <= 10) * (camera_route[:, :, 1] <= 10) * (camera_route[:, :, 2] >= 240)
        # camera_route_waypoint = np.expand_dims(np.array(camera_route_waypoint), axis=2)
        # camera_route_waypoint = resize(camera_route_waypoint, (self.obs_size, self.obs_size)) * 255
        # camera_route_surface = rgb_to_display_surface(camera_route, self.display_size)
        # self.display.blit(camera_route_surface, (self.display_size*2, 0))

        # Display camera_depth
        camera_depth_original = resize(self.camera_depth_img, (self.obs_size, self.obs_size)) * 255
        camera_depth_surface = rgb_to_display_surface(camera_depth_original, self.display_size)

        # self.display.blit(camera_depth_surface, (self.display_size*2, 0))
        # Display multi_fusion image
        camera_gray = self.rgb_to_gray(camera)
        camera_gray = np.expand_dims(camera_gray, axis=2)
        camera_gray_zeros = np.zeros_like(camera_gray)
        camera_gray_single = np.concatenate((camera_gray, camera_gray_zeros, camera_gray_zeros), axis=-1)
        camera_gray_single_surface = rgb_to_display_surface(camera_gray_single, self.display_size)
        # self.display.blit(camera_gray_single_surface, (self.display_size, 0))

        # camera_depth = self.rgb_to_gray(camera_depth_original)
        # camera_depth = np.expand_dims(camera_depth, axis=2)
        #
        # multi_fusion = np.concatenate((camera_gray, camera_depth, camera_route_waypoint ), axis=2)
        #
        # global multi_fusion_end
        # multi_fusion_end = Image.fromarray(multi_fusion.astype(np.uint8))
        # draw = ImageDraw.Draw(multi_fusion_end)
        # draw.rectangle([38, 95, 92, 128], fill=(30, 3, 2, 255))
        # del draw
        # multi_fusions = np.array(multi_fusion_end)
        # multi_fusions_surface = resize(multi_fusions, (self.obs_size, self.obs_size)) * 255
        # multi_fusion_surface = rgb_to_display_surface(multi_fusions_surface, self.display_size)
        # self.display.blit(multi_fusion_surface, (self.display_size, 0))
        # Display on pygame


        pygame.display.update()
        pygame.display.flip()

        # State observation
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw/180*np.pi


        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(np.cross(w,
                                       np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        v = self.ego.get_velocity()
        ego_vx = v.x
        ego_vy = v.y
        speed = np.sqrt(v.x**2 + v.y**2)
        state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])



        if self.pixor:
            # Vehicle classification and regression maps (requires further normalization)
            vh_clas = np.zeros((self.pixor_size, self.pixor_size))
            vh_regr = np.zeros((self.pixor_size, self.pixor_size, 8))

            for actor in self.world.get_actors().filter('vehicle.*'):
                x, y, yaw, l, w, vx, vy = get_info(actor)
                x_local, y_local, yaw_local, vx_local, vy_local = get_local_pose_and_velo(
                    (x, y, yaw, vx, vy), (ego_x, ego_y, ego_yaw, ego_vx, ego_vy))
                if actor.id != self.ego.id:
                    if abs(y_local) < self.obs_range / 2 + 1 and x_local < self.obs_range - self.d_behind + 1 and x_local > -self.d_behind - 1:

                        x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
                            local_info=(x_local, y_local, yaw_local, l, w),
                            d_behind=self.d_behind, obs_range=self.obs_range, image_size=self.pixor_size)
                        cos_t = np.cos(yaw_pixel)
                        sin_t = np.sin(yaw_pixel)
                        logw = np.log(w_pixel)
                        logl = np.log(l_pixel)
                        pixels = get_pixels_inside_vehicle(
                            pixel_info=(x_pixel, y_pixel,
                                        yaw_pixel, l_pixel, w_pixel),
                            pixel_grid=self.pixel_grid)
                        for pixel in pixels:
                            vh_clas[pixel[0], pixel[1]] = 1
                            dx = x_pixel - pixel[0]
                            dy = y_pixel - pixel[1]
                            vh_regr[pixel[0], pixel[1], :] = np.array(
                                [cos_t, sin_t, dx, dy, logw, logl, vx_local, vy_local])

            # Flip the image matrix so that the origin is at the left-bottom
            vh_clas = np.flip(vh_clas, axis=0)
            vh_regr = np.flip(vh_regr, axis=0)

            # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
            # pixor_state = [ego_x, ego_y, np.cos(
            #     ego_yaw), np.sin(ego_yaw), speed]
            pixor_state = [ego_x, ego_y, np.cos(
                ego_yaw), np.sin(ego_yaw), speed, ego_vx, ego_vy]

        obs = {
            'camera_gray': camera_gray_single.astype(np.uint8),
            'lidar_noground': lidar_noground.astype(np.uint8),
            'lidar': lidar.astype(np.uint8),
            'birdeye': birdeye.astype(np.uint8),
            'camera': camera.astype(np.uint8),
            # 'camera_route': camera_route.astype(np.uint8),
            'state': state,
            'camera_depth': camera_depth_original.astype(np.uint8),
            # 'multi_fusion': multi_fusions.astype(np.uint8),
            # 'semantic_image':semantic_image.astype(np.uint8)
        }

        if self.pixor:
            obs.update({
                'roadmap': roadmap.astype(np.uint8),
                'vh_clas': np.expand_dims(vh_clas, -1).astype(np.float32),
                'vh_regr': vh_regr.astype(np.float32),
                'pixor_state': pixor_state,
            })
        return obs

    def get_lidar(self):
        lidar = self.lidar
        lidars = lidar.transpose(2, 0, 1)
        return lidars



    def _get_reward(self, front_vehicle_t2c, around_vehicle_t2c, action, lateral_distance):
        """Calculate the step reward."""
        # reward for speed tracking
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        r_speed = -abs(speed - self.desired_speed)

        # reward for collision
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1

        # reward for steering:
        r_steer = -self.ego.get_control().steer ** 2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = 0
        if abs(dis) > self.out_lane_thres:
            r_out = -1

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        #cost for too fast
        r_fast = 0
        if lspeed_lon > self.desired_speed:
            r_fast = -1

        # cost for lateral acceleration
        r_lat = - abs(self.ego.get_control().steer) * lspeed_lon ** 2

        # cost for time to collision of the front vehicle
        front_max_time_to_collision = 3.0

        r_front_vehicle_t2c = 0
        if front_vehicle_t2c > 2 and front_vehicle_t2c <= front_max_time_to_collision:

            r_front_vehicle_t2c = -0.4
        elif front_vehicle_t2c > 1.2 and front_vehicle_t2c <= 2:
            r_front_vehicle_t2c = -0.8
        elif front_vehicle_t2c <= 1.2:
            r_front_vehicle_t2c = -1.0

        around_max_time_to_collision = 7.0
        r_around_vehicle_t2c = 0
        if abs(lateral_distance) > 1.2:
            r_around_vehicle_t2c = 0.0
        else:

            if  around_vehicle_t2c > 3.5 and around_vehicle_t2c < around_max_time_to_collision:
                r_around_vehicle_t2c = -0.4
            elif around_vehicle_t2c > 2.0 and around_vehicle_t2c <= 3.5:
                r_around_vehicle_t2c = -0.8
            elif around_vehicle_t2c <=2.0:
                r_around_vehicle_t2c = -1.5


        # print('r_attc',r_around_vehicle_t2c)
        # cost for smooth
        r_steer_differ = 0
        r_steer_differvalue = np.abs(action[1] - self.ego.get_control().steer)
        if r_steer_differvalue > 0.1 and r_steer_differvalue < 0.2:
            r_steer_differ = -1

        # cost for lateral distance
        r_lateral_distance = - abs(lateral_distance)



        r = 200 * r_collision + 200*r_front_vehicle_t2c+ 50*r_around_vehicle_t2c + 1 * lspeed_lon  + 2*r_fast+2* r_out + r_steer * 25 + 0.2 * r_lat - 0.1 + 2.0*r_steer_differ + 2.0*r_lateral_distance# default reward function
        #r = 200 * r_collision + 200 * r_front_vehicle_t2c  + 1 * lspeed_lon + 2 * r_fast + 2 * r_out + 0.2 * r_lat - 0.1
        self.total_reward += r
        # print('r_fast',r_fast)
        # print('r_lspeed_on',lspeed_lon)
        # print('r_collision',r_collision)
        # print('r_lat',r_lat)
        # print('r_fttc',front_vehicle_t2c)
        # print('r_out',r_out)
        # print('r_attc',r_around_vehicle_t2c)
        # print('total_reward',self.total_reward)


        return r


    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)

        # If collides
        if len(self.collision_hist) > 0:
            self.collision = 1


            return True

        # If reach maximum timestep
        if self.time_step > self.max_time_episode:
            return True

        # If at destination
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2) < 4:
                    return True

        v = self.ego.get_velocity()
        lspeed = np.array([v.x, v.y])
        ego_x, ego_y = get_pos(self.ego)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        lspeed_lon = np.dot(lspeed, w)

        # If out of lane
        if abs(dis) > self.out_lane_thres:
            self.out_lane_dis += lspeed_lon * self.dt
            if self.out_lane_dis > 10:
                return True
        else:
            self.out_lane_dis = 0

        # If blocked
        if lspeed_lon < 0.01:
            self.blocked_time += self.dt
            if self.blocked_time > 400: # defaut=50
                return True
        else:
            self.blocked_time = 0

        return False

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
                        actor.stop()
                    actor.destroy()
