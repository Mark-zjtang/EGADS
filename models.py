from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F
import torch.distributions
from torch.distributions.normal import Normal
from torch.distributions.transforms import Transform, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import numpy as np


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple, input_names=None, mask_names=None):
  global output
  if type(x_tuple) == dict:
    x_multi_obschannel, x_tuple_obschannel, x_sizes_obschannel  = {}, {}, {}
    for name in input_names:
      x_tuple_obschannel[name] = (x_tuple[name][1:],)
      x_sizes_obschannel[name] = tuple(map(lambda x: x.size(), x_tuple_obschannel[name]))
      x_multi_obschannel[name] = torch.as_tensor(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple_obschannel[name], x_sizes_obschannel[name])))
    y = f(x_multi_obschannel)
    y_size = y.size()
    output = y.view(x_sizes_obschannel[input_names[0]][0][0], x_sizes_obschannel[input_names[0]][0][1], *y_size[1:])
    return output
  else:
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    if type(y) == dict:
      y_size, multi_output= {}, {}
      for name in input_names+mask_names:
        y_size[name] = y[name].size()
        multi_output[name] = y[name].view(x_sizes[0][0], x_sizes[0][1], *y_size[name][1:])
      return multi_output
    else:
      y_size = y.size()
      output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
      return output


class TransitionModel(nn.Module):
  __constants__ = ['min_std_dev']

  def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.min_std_dev = min_std_dev
    self.fc_embed_state_action1 = nn.Linear(state_size + action_size, hidden_size)
    self.fc_embed_state_action2 = nn.Linear(hidden_size, hidden_size)
    self.fc_embed_state_action3 = nn.Linear(hidden_size, belief_size)
    self.rnn = nn.GRUCell(belief_size, belief_size)
    self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
    self.fc_state_prior1 = nn.Linear(hidden_size, hidden_size)
    self.fc_state_prior2 = nn.Linear(hidden_size, hidden_size)
    self.fc_state_prior3 = nn.Linear(hidden_size, 2 * state_size)
    self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
    self.fc_state_posterior1 = nn.Linear(hidden_size, hidden_size)
    self.fc_state_posterior2 = nn.Linear(hidden_size, hidden_size)
    self.fc_state_posterior3 = nn.Linear(hidden_size, 2 * state_size)
    self.modules = [self.fc_embed_state_action1, self.fc_embed_state_action2, self.fc_embed_state_action3, self.fc_embed_belief_prior, self.fc_state_prior1, self.fc_state_prior2, self.fc_state_prior3, self.fc_embed_belief_posterior, self.fc_state_posterior1, self.fc_state_posterior2, self.fc_state_posterior3]

  # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
  # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
  # t :  0  1  2  3  4  5
  # o :    -X--X--X--X--X-
  # a : -X--X--X--X--X-
  # n : -X--X--X--X--X-
  # pb: -X-
  # ps: -X-
  # b : -x--X--X--X--X--X-
  # s : -x--X--X--X--X--X-
  # @jit.script_method
  def forward(self, prev_state:torch.Tensor, actions:torch.Tensor, prev_belief:torch.Tensor, observations:Optional[torch.Tensor]=None, nonterminals:Optional[torch.Tensor]=None) -> List[torch.Tensor]:
    '''
    Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
    Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
            torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
    '''
    # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    T = actions.size(0) + 1
    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
    beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
    # Loop over time sequence
    for t in range(T - 1):
      _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
      _state = _state if nonterminals is None else _state * nonterminals[t]  # Mask if previous transition was terminal
      # Compute belief (deterministic hidden state)
      hidden = self.act_fn(self.fc_embed_state_action3(self.act_fn(self.fc_embed_state_action2(self.act_fn(self.fc_embed_state_action1(torch.cat([_state, actions[t]], dim=1)))))))
      beliefs[t + 1] = self.rnn(hidden, beliefs[t])
      # Compute state prior by applying transition dynamics
      hidden = self.act_fn(self.fc_state_prior2(self.act_fn(self.fc_state_prior1(self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))))))
      prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior3(hidden), 2, dim=1)
      prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
      prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
      if observations is not None:
        # Compute state posterior by applying transition dynamics and using current observation
        t_ = t - 1  # Use t_ to deal with different time indexing for observations
        hidden = self.act_fn(self.fc_state_posterior2(self.act_fn(self.fc_state_posterior1(self.act_fn(self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))))))
        posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior3(hidden), 2, dim=1)
        posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
        posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
    # Return new hidden states
    hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
    if observations is not None:
      hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
    return hidden


class SymbolicObservationModel(nn.Module):
  def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu', input_names=['lidar', 'camera'], mask_names=None):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.input_names = input_names
    self.mask_names = mask_names
    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, embedding_size)
    self.fc4 = nn.Linear(embedding_size, observation_size)
    self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]
    self.embedding_size = embedding_size

  # @jit.script_method
  def forward(self, belief, state):
    observation = {}
    for name in self.input_names:
      hiddens = torch.cat([belief, state], dim=1) # No nonlinearity here
      observation[name] = self.compute_hidden(hiddens)
    if len(self.mask_names) >=1:
      for name in self.mask_names:
        hiddens = torch.cat([belief, state], dim=1) # No nonlinearity here
        hidden = self.fc2(hiddens)
        hidden = self.fc3(hidden)
        observation[name] = self.compute_hidden(hidden)
    return observation

  def compute_hidden(self, hidden):
    hidden = self.act_fn(self.fc1(hidden))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    hidden = self.fc4(hidden)
    return hidden

class VisualObservationModel(nn.Module):
  __constants__ = ['embedding_size']
  
  def __init__(self, belief_size, state_size, embedding_size, activation_function='relu', input_names=['lidar', 'camera'], mask_names=['birdeye']):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
    self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
    self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
    self.conv4 = nn.ConvTranspose2d(32, 32, 6, stride=2)
    self.conv5 = nn.ConvTranspose2d(32, 3, 2, stride=2)
    self.input_names = input_names
    self.mask_names = mask_names
    self.generate_mask = False
    if len(self.mask_names) >= 1:
      self.generate_mask = True
    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, embedding_size)
    self.modules = [self.fc1, self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc1, self.fc2, self.fc3 ]
  # @jit.script_method
  def forward(self, belief, state):
    observation = {}

    hiddens = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
    observation[self.input_names[0]] = self.compute_hidden(hiddens)
    if len(self.input_names) > 1:
      hidden = self.act_fn(self.fc2(hiddens))
      hidden = self.act_fn(self.fc3(hidden))
      observation[self.input_names[1]] = self.compute_hidden(hidden)
      if len(self.input_names) > 2:
        observation[self.input_names[2]] = self.compute_hidden(hidden)

    if len(self.mask_names) >=1:
      for name in self.mask_names:
        hiddens = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
        hidden = self.fc2(hiddens)
        hidden = self.fc3(hidden)
        observation[name] = self.compute_hidden(hidden)
    return observation

  def compute_hidden(self, hidden):
    hidden = hidden.view(-1, self.embedding_size, 1, 1)
    hidden = self.act_fn(self.conv1(hidden))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    hidden = self.conv5(hidden)
    return hidden


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu',
                     input_names=['lidar', 'camera'], mask_names=['birdeye']):
  if symbolic:
    return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function,
                                    input_names, mask_names)
  else:
    return VisualObservationModel(belief_size, state_size, embedding_size, activation_function, input_names, mask_names)

class RewardModel(nn.Module):
  def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
    # [--belief-size: 200, --hidden-size: 200, --state-size: 30]
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)
    self.modules = [self.fc1, self.fc2, self.fc3]

  # @jit.script_method
  def forward(self, belief, state):
    x = torch.cat([belief, state],dim=1)
    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    reward = self.fc3(hidden).squeeze(dim=1)
    return reward

class ValueModel(nn.Module):
  def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, 1)
    self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

  # @jit.script_method
  def forward(self, belief, state):
    x = torch.cat([belief, state],dim=1)
    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    reward = self.fc4(hidden).squeeze(dim=1)
    return reward

class ActorModel(nn.Module):
  def __init__(self, belief_size, state_size, hidden_size, action_size, dist='tanh_normal',
                activation_function='elu', min_std=1e-4, init_std=5, mean_scale=5):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, hidden_size)
    self.fc5 = nn.Linear(hidden_size, 2*action_size)
    self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

    self._dist = dist
    self._min_std = min_std
    self._init_std = init_std
    self._mean_scale = mean_scale

  # @jit.script_method
  def forward(self, belief, state):
    raw_init_std = torch.log(torch.exp(torch.tensor(self._init_std, dtype=torch.float)) - 1)
    x = torch.cat([belief, state],dim=1)
    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    hidden = self.act_fn(self.fc4(hidden))
    action = self.fc5(hidden).squeeze(dim=1)

    action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
    action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
    action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
    return action_mean, action_std

  def get_action(self, belief, state, det=False):
    action_mean, action_std = self.forward(belief, state)
    dist = Normal(action_mean, action_std)
    dist = TransformedDistribution(dist, TanhBijector())
    dist = torch.distributions.Independent(dist,1)
    dist = SampleDist(dist)
    if det: return dist.mode()
    else: return dist.rsample()


class SymbolicEncoder(nn.Module):
  def __init__(self, observation_size, embedding_size, activation_function='relu', input_names=['lidar', 'camera'], device='cuda'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(observation_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, embedding_size)
    self.fc4 = nn.Linear(embedding_size, embedding_size)
    self.fc5 = nn.Linear(embedding_size, embedding_size)
    self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
    self.input_name = input_names
    self.device = device

  # @jit.script_method
  def forward(self, observation):
    hidden = {}
    for name in self.input_name:
      hidden[name] = self.act_fn(self.fc1(observation[name].to(device=self.device)))
      hidden[name] = self.act_fn(self.fc1(observation[name]))
      hidden[name] = self.act_fn(self.fc2(hidden[name]))
      hidden[name] = self.fc3(hidden[name])

    hiddens = torch.cat(list(hidden.values()), dim=-1)


    return hiddens


class VisualEncoder(nn.Module):
  __constants__ = ['embedding_size']
  
  def __init__(self, embedding_size, activation_function='relu', input_names=['lidar', 'camera'], device='cuda'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.conv0 = nn.Conv2d(3, 32, 2, stride=2)
    self.conv1 = nn.Conv2d(32, 32, 4, stride=2)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
    self.fc = nn.Identity()
    self.fc1 = nn.Linear(embedding_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size*len(input_names), embedding_size)
    self.fc4 = nn.Linear(embedding_size, embedding_size)
    self.modules = [self.conv0, self.conv1, self.conv2, self.conv3, self.conv4, self.fc1, self.fc2, self.fc3, self.fc4]
    self.input_name = input_names
    self.device = device

  # @jit.script_method
  def forward(self, observation):
    hidden = {}
    for name in self.input_name:
      hidden[name] = self.act_fn(self.conv0(observation[name].to(device=self.device)))
      hidden[name] = self.act_fn(self.conv1(hidden[name]))
      hidden[name] = self.act_fn(self.conv2(hidden[name]))
      hidden[name] = self.act_fn(self.conv3(hidden[name]))
      hidden[name] = self.act_fn(self.conv4(hidden[name]))
      hidden[name] = hidden[name].view(-1, 1024)
      hidden[name] = self.fc(hidden[name])  # Identity if embedding size is 1024 else linear projection

    hiddens = sum(hidden.values())

    return hiddens


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu', input_names=['lidar', 'camera'],device='cuda'):
  if symbolic:
    return SymbolicEncoder(observation_size, embedding_size, activation_function, input_names, device)
  else:
    return VisualEncoder(embedding_size, activation_function, input_names, device)
# "atanh", "TanhBijector" and "SampleDist" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))

class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = torch.distributions.constraints.real
        self.codomain = torch.distributions.constraints.interval(-1.0, 1.0)

    @property
    def sign(self): return 1.

    def _call(self, x): return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997),
            y)
        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))


class SampleDist:
  def __init__(self, dist, samples=100):
    self._dist = dist
    self._samples = samples

  @property
  def name(self):
    return 'SampleDist'

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def mean(self):
    sample = dist.rsample()
    return torch.mean(sample, 0)

  def mode(self):
    dist = self._dist.expand((self._samples, *self._dist.batch_shape))
    sample = dist.rsample()
    logprob = dist.log_prob(sample)
    batch_size = sample.size(1)
    feature_size = sample.size(2)
    indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
    return torch.gather(sample, 0, indices).squeeze(0)

  def entropy(self):
    dist = self._dist.expand((self._samples, *self._dist.batch_shape))
    sample = dist.rsample()
    logprob = dist.log_prob(sample)
    return -torch.mean(logprob, 0)

  def sample(self):
    return self._dist.sample()
