import numpy as np
import torch
from env_carla_obschannel import postprocess_observation, preprocess_observation_


class ExperienceReplay():
  def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device, input_names):
    self.device = device
    self.symbolic_env = symbolic_env
    self.size = size
    # self.observations = np.empty((size, observation_size) if symbolic_env else (size, 3, 64, 64), dtype=np.float32 if symbolic_env else np.uint8)
    self.observations = {}
    self.input_names = input_names
    for name in self.input_names:
      self.observations[name] = np.empty((size, observation_size) if symbolic_env else (size, 3, 128, 128), dtype=np.float32 if symbolic_env else np.uint8)
    self.actions = np.empty((size, action_size), dtype=np.float32)
    self.rewards = np.empty((size, ), dtype=np.float32) 
    self.nonterminals = np.empty((size, 1), dtype=np.float32)
    self.idx = 0
    self.full = False  # Tracks if memory has been filled/all slots are valid
    self.steps, self.episodes = 0, 0  # Tracks hsow much experience has been used in total
    self.bit_depth = bit_depth

  def append(self, observation, action, reward, done):
    for name in self.input_names:
      if self.symbolic_env:
        self.observations[name][self.idx] = observation[name].numpy()
      else:
        self.observations[name][self.idx] = postprocess_observation(observation[name].numpy(), self.bit_depth)  # Decentre and discretise visual observations (to save memory)
    self.actions[self.idx] = action.numpy()
    self.rewards[self.idx] = reward
    self.nonterminals[self.idx] = not done
    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0
    self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

  # Returns an index for a valid single sequence chunk uniformly sampled from the memory
  def _sample_idx(self, L):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - L)
      idxs = np.arange(idx, idx + L) % self.size
      valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
    return idxs

  def _retrieve_batch(self, idxs, n, L):
    vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
    observations = {}
    multi_observations = {}

    for name in self.input_names:
      observations[name] = torch.as_tensor(self.observations[name][vec_idxs].astype(np.float32))
      if not self.symbolic_env:
        preprocess_observation_(observations[name], self.bit_depth)  # Undo discretisation for visual observations
      multi_observations[name] = torch.as_tensor(observations[name].reshape(L, n, *observations[name].shape[1:])).to(device=self.device)
    return multi_observations, self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1)

  # Returns a batch of sequence chunks uniformly sampled from the memory
  def sample(self, n, L):
    batch = self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(n)]), n, L)
    # [1578 1579 1580 ... 1625 1626 1627]                                                                                                                                        | 0/100 [00:00<?, ?it/s]
    # [1049 1050 1051 ... 1096 1097 1098]
    # [1236 1237 1238 ... 1283 1284 1285]
    # ...
    # [2199 2200 2201 ... 2246 2247 2248]
    # [ 686  687  688 ...  733  734  735]
    # [1377 1378 1379 ... 1424 1425 1426]]
    return [batch[0], torch.as_tensor(batch[1]).to(device=self.device), torch.as_tensor(batch[2]).to(device=self.device),  torch.as_tensor(batch[3]).to(device=self.device)]
