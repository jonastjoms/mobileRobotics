#!/usr/bin/env python
from __future__ import division
from __future__ import absolute_import
import copy
import torch
from torch import nn
from torch.distributions import Distribution, Normal
from itertools import izip


class Actor(nn.Module):
  def __init__(self, hidden_size, stochastic=True, layer_norm=False):
    super(Actor, self).__init__()
    layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
    if layer_norm:
      layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]  # Insert layer normalisation between fully-connected layers and nonlinearities
    self.policy = nn.Sequential(*layers)
    if stochastic:
      self.policy_log_std = nn.Parameter(torch.tensor([[0.]]))

  def forward(self, state):
    policy = self.policy(state)
    return policy


class TanhNormal(Distribution):
  def __init__(self, loc, scale):
    super(TanhNormal, self).__init__()
    self.normal = Normal(loc, scale)

  def sample(self):
    return torch.tanh(self.normal.sample())

  def rsample(self):
    return torch.tanh(self.normal.rsample())

  def rsample_log_prob(self):
    value = self.normal.rsample()
    log_prob = self.normal.log_prob(value)
    value = torch.tanh(value)
    log_prob -= torch.log1p(-value.pow(2) + 1e-6)
    return value, log_prob.sum(dim=1)

  # Calculates log probability of value using the change-of-variables technique (uses log1p = log(1 + x) for extra numerical stability)
  def log_prob(self, value):
    inv_value = (torch.log1p(value) - torch.log1p(-value)) / 2  # artanh(y)
    return self.normal.log_prob(inv_value) - torch.log1p(-value.pow(2) + 1e-6)  # log p(f^-1(y)) + log |det(J(f^-1(y)))|

  @property
  def mean(self):
    return torch.tanh(self.normal.mean)


class SoftActor(nn.Module):
  def __init__(self, hidden_size):
    super(SoftActor, self).__init__()
    self.log_std_min, self.log_std_max = -20, 2  # Constrain range of standard deviations to prevent very deterministic/stochastic policies
    layers = [nn.Linear(8, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 3)]
    self.policy = nn.Sequential(*layers)

  def forward(self, state):
    policy_mean, policy_log_std = self.policy(state).chunk(2, dim=1)
    policy_log_std = torch.clamp(policy_log_std, min=self.log_std_min, max=self.log_std_max)
    policy = TanhNormal(policy_mean, policy_log_std.exp())
    return policy


class Critic(nn.Module):
  def __init__(self, hidden_size, state_action=False, layer_norm=False):
    super(Critic, self).__init__()
    self.state_action = state_action
    layers = [nn.Linear(8 + (2 if state_action else 0), hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
    if layer_norm:
      layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]  # Insert layer normalisation between fully-connected layers and nonlinearities
    self.value = nn.Sequential(*layers)

  def forward(self, state, action=None):
    if self.state_action:
      value = self.value(torch.cat([state, action], dim=1))
    else:
      value = self.value(state)
    return value.squeeze(dim=1)


class ActorCritic(nn.Module):
  def __init__(self, hidden_size):
    super(ActorCritic, self).__init__()
    self.actor = Actor(hidden_size, stochastic=True)
    self.critic = Critic(hidden_size)

  def forward(self, state):
    policy = Normal(self.actor(state), self.actor.policy_log_std.exp())
    value = self.critic(state)
    return policy, value


class DQN(nn.Module):
  def __init__(self, hidden_size, num_actions=5):
    super(DQN, self).__init__()
    layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, num_actions)]
    self.dqn = nn.Sequential(*layers)

  def forward(self, state):
    values = self.dqn(state)
    return values


def create_target_network(network):
  target_network = copy.deepcopy(network)
  for param in target_network.parameters():
    param.requires_grad = False
  return target_network


def update_target_network(network, target_network, polyak_factor):
  for param, target_param in izip(network.parameters(), target_network.parameters()):
    target_param.data = polyak_factor * target_param.data + (1 - polyak_factor) * param.data
