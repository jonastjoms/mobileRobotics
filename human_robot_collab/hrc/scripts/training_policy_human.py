#!/usr/bin/env python
# license removed for brevity
from __future__ import with_statement
from __future__ import absolute_import
import roslib
roslib.load_manifest('hrc_msgs')
import rospy
import os
import roslaunch
import sys
import copy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from hrc_msgs.msg import observations
import tf
import numpy as np
from math import pi, sqrt
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from collections import deque
import random
import torch
import time
import cPickle as pickle
from torch import optim
from tqdm import tqdm
from hyperparams_ur10 import OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, TEST_INTERVAL, UPDATE_INTERVAL, UPDATE_START, SAVE_INTERVAL
from models_ur10_human import Critic, SoftActor, create_target_network, update_target_network
from decimal import Decimal
import rospkg

rospack = rospkg.RosPack()
package_path = rospack.get_path("hrc")

print("Training with saved transitions")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
done = False
reward = 0

# SAC initialisations
action_space = 2
state_space = 8
actor = SoftActor(HIDDEN_SIZE).to(device)
critic_1 = Critic(HIDDEN_SIZE, state_action=True).to(device)
critic_2 = Critic(HIDDEN_SIZE, state_action=True).to(device)
value_critic = Critic(HIDDEN_SIZE).to(device)
# Load models
if os.path.isfile(package_path+"/scripts/checkpoints_human/ali_random.pth"):
    print("Loading models")
    checkpoint = torch.load(package_path+"/scripts/checkpoints_human/ali_random.pth")
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
    critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
    value_critic.load_state_dict(checkpoint['value_critic_state_dict'])
    UPDATE_START = 1
else:
    print("No checkpoint found at '{}'".format(package_path+"/scripts/checkpoints_human/ali_random.pth"))

target_value_critic = create_target_network(value_critic).to(device)
actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LEARNING_RATE)
value_critic_optimiser = optim.Adam(value_critic.parameters(), lr=LEARNING_RATE)
D = deque(maxlen=REPLAY_SIZE)
# Automatic entropy tuning init
target_entropy = -np.prod(action_space).item()
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optimizer = optim.Adam([log_alpha], lr=LEARNING_RATE)

# Load models
if os.path.isfile(package_path+"/scripts/checkpoints_human/ali_random.pth"):
    target_value_critic.load_state_dict(checkpoint['target_value_critic_state_dict'])
    actor_optimiser.load_state_dict(checkpoint['actor_optimiser_state_dict'])
    critics_optimiser.load_state_dict(checkpoint['critics_optimiser_state_dict'])
    value_critic_optimiser.load_state_dict(checkpoint['value_critic_optimiser_state_dict'])
    alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
    D = pickle.load( open( package_path+"/scripts/checkpoints_human/ali_random.p", "rb" ) )

pbar = tqdm(xrange(1, 30000 + 1), unit_scale=1, smoothing=0)

# Training loop
for steps in pbar:
    try:
        # Randomly sample a batch of transitions B = {(s, a, r, s', d)} from D
        batch = random.sample(D, BATCH_SIZE)
        batch = dict((k, torch.cat([d[k] for d in batch], dim=0)) for k in batch[0].keys())

        # Compute targets for Q and V functions
        y_q = batch['reward'] + DISCOUNT * (1 - batch['done']) * target_value_critic(batch['next_state'])
        policy = actor(batch['state'])
        action_update, log_prob = policy.rsample_log_prob()  # a(s) is a sample from mu(:|s) which is differentiable wrt theta via the reparameterisation trick
        # Automatic entropy tuning
        alpha_loss = -(log_alpha.float() * (log_prob + target_entropy).float().detach()).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
        alpha = log_alpha.exp()
        weighted_sample_entropy = (alpha.float() * log_prob).view(-1,1)

        # Weighted_sample_entropy = ENTROPY_WEIGHT * policy.log_prob(action).sum(dim=1)  # Note: in practice it is more numerically stable to calculate the log probability when sampling an action to avoid inverting tanh
        y_v = torch.min(critic_1(batch['state'], action_update.detach()), critic_2(batch['state'], action_update.detach())) - weighted_sample_entropy.detach()

        # Update Q-functions by one step of gradient descent
        value_loss = (critic_1(batch['state'], batch['action']) - y_q).pow(2).mean() + (critic_2(batch['state'], batch['action']) - y_q).pow(2).mean()
        critics_optimiser.zero_grad()
        value_loss.backward()
        critics_optimiser.step()

        # Update V-function by one step of gradient descent
        value_loss = (value_critic(batch['state']) - y_v).pow(2).mean()
        value_critic_optimiser.zero_grad()
        value_loss.backward()
        value_critic_optimiser.step()

        # Update policy by one step of gradient ascent
        policy_loss = (weighted_sample_entropy - critic_1(batch['state'], action_update)).mean()
        actor_optimiser.zero_grad()
        policy_loss.backward()
        actor_optimiser.step()

        # Update target value network
        update_target_network(value_critic, target_value_critic, POLYAK_FACTOR)

    except KeyboardInterrupt:
        break

print("Training finished, saving models")
torch.save({
'actor_state_dict': actor.state_dict(),
'critic_1_state_dict': critic_1.state_dict(),
'critic_2_state_dict': critic_1.state_dict(),
'value_critic_state_dict': value_critic.state_dict(),
'target_value_critic_state_dict': target_value_critic.state_dict(),
'value_critic_optimiser_state_dict': value_critic_optimiser.state_dict(),
'actor_optimiser_state_dict': actor_optimiser.state_dict(),
'critics_optimiser_state_dict': critics_optimiser.state_dict(),
'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
},package_path+"/scripts/checkpoints_human/ali_random.pth")
print("Saving replay buffer")
pickle.dump( D, open( package_path+"/scripts/checkpoints_human/ali_random.p", "wb" ) )
