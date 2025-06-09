from gym.envs.registration import register

register(
    id='carla-v1',
    entry_point='gym_carla.envs:CarlaEnv',
)