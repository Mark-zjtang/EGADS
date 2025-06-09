from typing import Optional, List
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution

from nde import transforms
from nde.resnet import ResidualNet, MLPNet
import nde.utils as utils
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
      y_size, multi_output = {}, {}
      for name in input_names+mask_names:
        y_size[name] = y[name].size()
        multi_output[name] = y[name].view(x_sizes[0][0], x_sizes[0][1], *y_size[name][1:])
      return multi_output
    else:
      y_size = y.size()
      output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
      return output

# jit.ScriptModule
class TransitionModel(nn.Module):
  __constants__ = ['min_std_dev']

  def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, config, activation_function='relu', input_names=['lidar','camera'], min_std_dev=0.1):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.min_std_dev = min_std_dev
    self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
    self.rnn = nn.GRUCell(belief_size, belief_size)
    self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
    self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
    self.input_names = input_names
    self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
    self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

    # self.fc_embed_belief_flow_prior_context1 =nn.Linear(belief_size, hidden_size)
    # self.fc_embed_belief_flow_prior_context2 =nn.Linear(hidden_size, 2 * state_size)
    # self.fc_embed_belief_flow_posterior_context1 =nn.Linear(belief_size + embedding_size, hidden_size)
    # self.fc_embed_belief_flow_posterior_context2 =nn.Linear(hidden_size, 2 * state_size)

    self.config = config
    self.state_size = state_size
    self.belief_size = belief_size
    self.embedding_size = embedding_size
    self.prior_transform = self.create_prior(config)
    if config.approximate_posterior_type == 'same-as-prior':
      print("Posterior same as prior")
      self.approximate_posterior_transform = self.prior_transform
    else:
      self.approximate_posterior_transform = self.create_approximate_posterior(config)
    self._log_z = 0.5 * np.prod((state_size,)) * np.log(2 * np.pi)

    # self.modules = [self.fc_embed_state_action, self.fc_embed_belief_prior, self.fc_state_prior, self.fc_embed_belief_posterior, self.fc_state_posterior]
    self.modules = [self.fc_embed_state_action, self.fc_embed_belief_prior, self.fc_state_prior, self.fc_embed_belief_posterior, self.fc_state_posterior]
    if self.prior_transform:
      self.modules += [self.prior_transform,]

    if (not (config.approximate_posterior_type == 'same-as-prior')) and self.approximate_posterior_transform:
      self.modules += [self.approximate_posterior_transform,]

  def create_prior(self, config):
    if config.prior_type == 'standard-normal':
      transform = None 
    else:
      transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            self.create_linear_transform(config),
            self.create_base_transform(config, i, post=False, context_features=self.belief_size)
        ]) for i in range(config.num_flow_steps)
      ])
      transform = transforms.CompositeTransform([
        transform,
        self.create_linear_transform(config)
      ])
    return transform
  
  def create_approximate_posterior(self, config):
    if config.approximate_posterior_type == 'diagonal-normal':
      transform = None 
    else:
      transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            self.create_linear_transform(config),
            self.create_base_transform(config, i, post=True, context_features=self.belief_size)
        ]) for i in range(config.num_flow_steps)
      ])
      transform = transforms.CompositeTransform([
        transform,
        self.create_linear_transform(config)
      ])
    return transform
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
    prior_contexts, posterior_contexts = [torch.empty(0)] * T, [torch.empty(0)] * T
    beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state

    # Loop over time sequence
    for t in range(T - 1):
      _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
      _state = _state if nonterminals is None else _state * nonterminals[t]  # Mask if previous transition was terminal
      # Compute belief (deterministic hidden state)
      hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
      beliefs[t + 1] = self.rnn(hidden, beliefs[t])
      # Compute state prior by applying transition dynamics
      hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))

      prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
      prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev

      if self.config.fix_prior:
        prior_means[t + 1] = torch.zeros_like(prior_means[t + 1])
        prior_std_devs[t + 1] = torch.ones_like(prior_std_devs[t + 1])
      
      # Prior Base Samples
      prior_states_ = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
      # h = self.act_fn(self.fc_embed_belief_flow_prior_context1(beliefs[t + 1]))
      # prior_contexts[t + 1] = self.fc_embed_belief_flow_prior_context2(h)
      prior_contexts[t + 1] = beliefs[t + 1]
      prior_states[t + 1] = self._transfer_from_base(prior_states_, prior_contexts[t + 1], is_prior=True, freeze_context=self.config.freezecontext)  # transfered samples
      
      if observations is not None:
        # Compute state posterior by applying transition dynamics and using current observation
        t_ = t - 1  # Use t_ to deal with different time indexing for observations
        # Choice A. use MLP embedding
          # h = self.act_fn(self.fc_embed_belief_flow_posterior_context1(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
          # posterior_contexts[t + 1] = self.fc_embed_belief_flow_posterior_context2(h)
        # Choice B. use [belief + observation]
          # posterior_contexts[t + 1] = torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)
        # Choice C. only use belief
        posterior_contexts[t + 1] = beliefs[t + 1]

        hidden = self.act_fn(self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
        posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
        posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev

        # Posterior Base Samples
        posterior_states_ = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
        posterior_states[t + 1] = self._transfer_from_base(posterior_states_, posterior_contexts[t + 1], is_prior=False, freeze_context=self.config.freezecontext)

    # === Flow part ===
    # beliefs are the context
    beliefs = torch.stack(beliefs[1:], dim=0)  # contexts
    prior_states = torch.stack(prior_states[1:], dim=0)  # base samples
    prior_means = torch.stack(prior_means[1:], dim=0) 
    prior_std_devs = torch.stack(prior_std_devs[1:], dim=0)
    prior_contexts = torch.stack(prior_contexts[1:], dim=0)
    
    # Return new hidden states
    hidden = [beliefs, prior_states, prior_means, prior_std_devs, prior_contexts]
    if observations is not None:
      posterior_states = torch.stack(posterior_states[1:], dim=0)
      posterior_means = torch.stack(posterior_means[1:], dim=0)
      posterior_std_devs = torch.stack(posterior_std_devs[1:], dim=0)      
      posterior_contexts = torch.stack(posterior_contexts[1:], dim=0)

      hidden += [posterior_states, posterior_means, posterior_std_devs, posterior_contexts]
    return hidden

  def _transfer_from_base(self, base_samples, contexts_, is_prior, freeze_context=False):
    if is_prior:
      transform = self.prior_transform
    else:
      transform = self.approximate_posterior_transform
    
    # base_samples: [49(T), 50(B), 30(Stoch size)]
    # contexts: [49(T), 50(B), 200(Deter size)]

    if freeze_context:
      with torch.no_grad():
        contexts = contexts_.detach()
    else:
      contexts = contexts_

    if transform is None:
      samples = base_samples
    else:
      sample_shape_backup = base_samples.shape
      base_samples = base_samples.view(-1, base_samples.shape[-1])
      contexts = contexts.view(-1, contexts.shape[-1])
      samples, _ = transform.inverse(base_samples, context=contexts)
      samples = samples.view(sample_shape_backup)

    return samples

  def _log_prob_gaussian(self, x, mean, std):
    # Compute log prob(x) of N(mean, std).
    norm_inputs = (x - mean) / std
    log_prob = -0.5 * self.sum_except_batch(norm_inputs ** 2, num_batch_dims=2)
    log_prob -= self.sum_except_batch(torch.log(std), num_batch_dims=2)
    log_prob -= self._log_z
    return log_prob

  def _kl_divergence(self, prior_mean, prior_std_dev, prior_context, posterior_mean, posterior_std_dev, posterior_context):
    post_base_sample = posterior_mean + posterior_std_dev * torch.randn_like(posterior_mean)
    log_prob_base = self._log_prob_gaussian(post_base_sample, posterior_mean, posterior_std_dev)  # [50, 50] -> [2500,]
    
    shape_backup_sample = post_base_sample.shape
    shape_backup_log_prob = log_prob_base.shape

    base_sample = post_base_sample.view(-1, self.state_size)
    log_prob_base = log_prob_base.view(-1,)
    posterior_context = posterior_context.view(-1, posterior_context.shape[-1])
    prior_context = prior_context.view(-1, prior_context.shape[-1])

    if self.approximate_posterior_transform is None:
      sample = base_sample
      logabsdet = torch.zeros_like(log_prob_base)
    else:
      sample, logabsdet = self.approximate_posterior_transform.inverse(base_sample, context=posterior_context)

    log_q_z = log_prob_base - logabsdet
    log_q_z = log_q_z.view(shape_backup_log_prob)

    if self.prior_transform is None:
      prior_base_sample = sample 
      logabsdet = torch.zeros_like(log_prob_base)
    else:
      prior_base_sample, logabsdet = self.prior_transform.forward(sample, context=prior_context)
    
    prior_base_sample = prior_base_sample.view(shape_backup_sample[0], shape_backup_sample[1], -1)

    log_prob = self._log_prob_gaussian(prior_base_sample, prior_mean, prior_std_dev)
    logabsdet = logabsdet.view(shape_backup_log_prob)
    log_p_z = log_prob + logabsdet
    
    return log_q_z - log_p_z

  def sum_except_batch(self, x, num_batch_dims):
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)

  def create_linear_transform(self, config):
    if config.linear_type == 'lu':
      return transforms.CompositeTransform([
        transforms.RandomPermutation(self.state_size),
        transforms.LULinear(self.state_size, identity_init=True)
      ])
    elif config.linear_type == 'svd':
      return transforms.SVDLinear(self.state_size, num_householder=4,
                                  identity_init=True)
    elif config.linear_type == 'perm':
      return transforms.RandomPermutation(self.state_size)
    else:
      raise ValueError

  def create_base_transform(self, config, i, post, context_features=None):
    if post:
      base_transform_type = config.approximate_posterior_type
    else:
      base_transform_type = config.prior_type
    
    # to be done
    if config.usemlp:
      print("Use MLP as parameter network")
      param_net = MLPNet
    else:
      print("Use Residual Network as parameter network")
      param_net = ResidualNet
    
    if base_transform_type == 'affine-coupling':
      return transforms.AffineCouplingTransform(
          mask=utils.create_alternating_binary_mask(
              features=self.state_size,
              even=(i % 2 == 0)
          ),
          transform_net_create_fn=lambda in_features,
                                          out_features: param_net(
              in_features=in_features,
              out_features=out_features,
              hidden_features=config.hidden_features,
              context_features=context_features,
              num_blocks=config.num_transform_blocks,
              activation=F.relu,
              dropout_probability=config.dropout_probability,
              use_batch_norm=config.use_batch_norm
          )
      )
    elif base_transform_type == 'rq-coupling':
      return transforms.PiecewiseRationalQuadraticCouplingTransform(
          mask=utils.create_alternating_binary_mask(
              features=self.state_size,
              even=(i % 2 == 0)
          ),
          transform_net_create_fn=lambda in_features,
                                          out_features: param_net(
              in_features=in_features,
              out_features=out_features,
              hidden_features=config.hidden_features,
              context_features=context_features,
              num_blocks=config.num_transform_blocks,
              activation=F.relu,
              dropout_probability=config.dropout_probability,
              use_batch_norm=config.use_batch_norm
          ),
          num_bins=config.num_bins,
          tails='linear',
          tail_bound=config.tail_bound,
          apply_unconditional_transform=config.apply_unconditional_transform,
      )
    elif base_transform_type == 'affine-autoregressive':
      return transforms.MaskedAffineAutoregressiveTransform(
          features=self.state_size,
          hidden_features=config.hidden_features,
          context_features=context_features,
          num_blocks=config.num_transform_blocks,
          use_residual_blocks=True,
          random_mask=False,
          activation=F.relu,
          dropout_probability=config.dropout_probability,
          use_batch_norm=config.use_batch_norm
      )
    elif base_transform_type == 'rq-autoregressive':
      return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
          features=self.state_size,
          hidden_features=config.hidden_features,
          context_features=context_features,
          num_bins=config.num_bins,
          tails='linear',
          tail_bound=config.tail_bound,
          num_blocks=config.num_transform_blocks,
          use_residual_blocks=True,
          random_mask=False,
          activation=F.relu,
          dropout_probability=config.dropout_probability,
          use_batch_norm=config.use_batch_norm
      )
    else:
      raise ValueError

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


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu', input_names=['lidar', 'camera'], mask_names=['birdeye']):
  if symbolic:
    return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function, input_names, mask_names)
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

class Behavior_cloningModel(nn.Module):
  def __init__(self, belief_size, state_size, hidden_size, action_size, dist='tanh_normal',
               activation_function='elu', min_std=1e-4, init_std=5, mean_scale=5):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, hidden_size)
    self.fc5 = nn.Linear(hidden_size, 2 * action_size)
    self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
    self.action_size = action_size

    self._dist = dist
    self._min_std = min_std
    self._init_std = init_std
    self._mean_scale = mean_scale
    self.log_z = 0.5 * np.prod((action_size,)) * np.log(2 * np.pi)

  # @jit.script_method
  def forward(self, belief, state):
    raw_init_std = torch.log(torch.exp(torch.tensor(self._init_std, dtype=torch.float)) - 1)
    x = torch.cat([belief, state], dim=1)
    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    hidden = self.act_fn(self.fc4(hidden))
    action = self.fc5(hidden).squeeze(dim=1)

    action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
    action_mean = self._mean_scale * torch.tanh(action_mean / self._mean_scale)
    action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std


    return action_mean, action_std, action

  def sum_except_batch(self, x, num_batch_dims):
    reduce_dims = list(range(num_batch_dims, 1))
    return torch.sum(x, dim=reduce_dims)

  def log_prob_gaussian(self, x, mean, std):
    # Compute log prob(x) of N(mean, std).
    norm_inputs = (x - mean) / std
    log_prob = -0.5 * self.sum_except_batch(norm_inputs ** 2, num_batch_dims=0)
    log_prob -= self.sum_except_batch(torch.log(std), num_batch_dims=0)
    log_prob -= self.log_z
    return log_prob

  def get_action(self, belief, state, expert_action, det=False, get_entropy=False,expert_train=True):
    action_mean, action_std, action = self.forward(belief, state)
    dist = Normal(action_mean, action_std)
    dist = TransformedDistribution(dist, TanhBijector())
    dist = torch.distributions.Independent(dist, 1)
    dist = SampleDist(dist)
    if expert_train:
      log_prob = self.log_prob_gaussian(expert_action, action_mean, action_std)
      log_prob = torch.sum(log_prob)
    else:
      log_prob = None
    if det: return dist.mode()
    elif get_entropy: return dist.rsample(), dist.entropy()
    else: return dist.rsample(), log_prob

  # def get_action_eval(self, belief, state, det=False, get_entropy=False):
  #   action_mean, action_std,_ = self.forward(belief, state)
  #   dist = Normal(action_mean, action_std)
  #   dist = TransformedDistribution(dist, TanhBijector())
  #   dist = torch.distributions.Independent(dist, 1)
  #   dist = SampleDist(dist)
  #   if det: return dist.mode()
  #   elif get_entropy: return dist.rsample(), dist.entropy()
  #   else: return dist.rsample()

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

  def get_action(self, belief, state, det=False, get_entropy=False):
    action_mean, action_std = self.forward(belief, state)
    dist = Normal(action_mean, action_std)
    dist = TransformedDistribution(dist, TanhBijector())
    dist = torch.distributions.Independent(dist,1)
    dist = SampleDist(dist)
    if det: return dist.mode()
    elif get_entropy: return dist.rsample(), dist.entropy()
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


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu', input_names=['lidar', 'camera'], device='cuda'):
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
