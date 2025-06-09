
import os 
os.environ['MUJOCO_GL'] = 'egl'
import numpy as np 
from dm_control import suite

from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
env = Env("cheetah-run", False, 64, 1000, 2, 5)

from IPython import embed; embed()
exit(1)

seed = 878
env = suite.load(domain_name='cheetah', task_name="run", task_kwargs={'random': seed})
print(env.reset().observation)

spec = env.action_spec()
act_list = []
for i in range(100):
  act = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
  act_list.append(act)

env = suite.load(domain_name='cheetah', task_name="run", task_kwargs={'random': seed})
print(env.reset().observation)

state_list_1 = []
reward_list_1 = []
done_list_1 = []

for a in act_list:
  state = env.step(a)
  state_list_1.append(state.observation)
  reward_list_1.append(state.reward)
  done_list_1.append(state.last())
  if state.last():
    break

# Reload with the same seed
env = suite.load(domain_name='cheetah', task_name="run", task_kwargs={'random': seed})
print(env.reset().observation)  # Same as first
# OrderedDict([('position', array([-0.02509198,  0.99953037,  0.03064377])),
#              ('velocity', array([ 0.00647689,  0.0152303 ]))])
state_list_2 = []
reward_list_2 = []
done_list_2 = []

for a in act_list:
  state = env.step(a)
  state_list_2.append(state.observation)
  reward_list_2.append(state.reward)
  done_list_2.append(state.last())
  if state.last():
    break

from IPython import embed; embed()
