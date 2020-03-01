from __future__ import with_statement
from __future__ import absolute_import
import os
import sys
import copy
import numpy as np
from math import pi, sqrt
from collections import deque
import random
import torch
import time
import cPickle as pickle
from torch import optim
from tqdm import tqdm
from hyperparams_ur10 import OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, UPDATE_INTERVAL, UPDATE_START, SAVE_INTERVAL
from models_ur10_agent import Critic, SoftActor, create_target_network, update_target_network
from decimal import Decimal


# Observation class
class Observation:

    def __init__(self):
        self.robot_x = 0
        self.robot_y = 0
        self.path_points = 0
        self.state = [self.robot_x, self.robot_y, self.path_points]

# Method to get state from observations
    def get_state(self, obs):
        self.robot_x = obs.robot_x
        self.robot_y = obs.robot_y
        self.path_points = obs.path_points
        self.state = [self.robot_x, self.robot_y, self.path_points]


def reset(start_pos):
    # Set robot position to start_pos
    return True

obs = Observation()

# Define reset/initial pose:
start_pos = np.array(0, 0, 0)
reset(start_pos)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
state = torch.tensor(observation.state).to(device)
done = False
reward = 0

# SAC initialisations
action_space = 2
state_space = 3
actor = SoftActor(HIDDEN_SIZE).to(device)
critic_1 = Critic(HIDDEN_SIZE, state_action=True).to(device)
critic_2 = Critic(HIDDEN_SIZE, state_action=True).to(device)
value_critic = Critic(HIDDEN_SIZE).to(device)

target_value_critic = create_target_network(value_critic).to(device)
actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LEARNING_RATE)
value_critic_optimiser = optim.Adam(value_critic.parameters(), lr=LEARNING_RATE)
D = deque(maxlen=REPLAY_SIZE)
# Automatic entropy tuning init
target_entropy = -np.prod(action_space).item()
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optimizer = optim.Adam([log_alpha], lr=LEARNING_RATE)

# Other variables
reward_sparse = False
reward_dense = True
# Set goal position, maybe every trial? Or every 1000 step?
pbar = tqdm(xrange(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)
first_update = True
reset_number = 0

# Training loop
for step in pbar:
    try:
      with torch.no_grad():

        # Get position of robot and path points here:
        obs.robot_x =
        state = torch.tensor(observation.state).to(device)
        if step < UPDATE_START:
          # To improve exploration take actions sampled from a uniform random distribution over actions at the start of training
          action = torch.tensor([2 * random.random() - 1, 2 * random.random() - 1], device=device).unsqueeze(0)
        else:
          # Observe state s and select action a ~ mu(a|s)
          action = actor(state.unsqueeze(0)).sample()
        # Execute a in the environment and observe next state s', reward r, and done signal d to indicate whether s' is terminal
        # Must compensate for translation
        current_pose = move_group.get_current_pose().pose
        current_rpy = move_group.get_current_rpy()
        total_action = [action[0][0], action[0][1]]
        #print(current_rpy)
        # Setting linear commands to be published
        training_msg.twist.linear.x = pose_goal.position.x - current_pose.position.x
        training_msg.twist.linear.y = pose_goal.position.y - current_pose.position.y
        training_msg.twist.linear.z = pose_goal.position.z - current_pose.position.z
        training_msg.twist.angular.z = 0 - current_rpy[2]
        # Setting angular commands, enforcing constraints
        if abs(current_rpy[0]) > abs_max_theta:
            if current_rpy[0] < 0 and total_action[0] < 0:
                training_msg.twist.angular.x = 0
            elif current_rpy[0] > 0 and total_action[0] > 0:
                training_msg.twist.angular.x = 0
            else:
                training_msg.twist.angular.x = total_action[0]
        else:
            training_msg.twist.angular.x = total_action[0]
        if abs(current_rpy[1]) > abs_max_phi:
            if current_rpy[1] < 0 and total_action[1] < 0:
                training_msg.twist.angular.y = 0
            elif current_rpy[1] > 0 and total_action[1] > 0:
                training_msg.twist.angular.y = 0
            else:
                training_msg.twist.angular.y = total_action[1]
        else:
            training_msg.twist.angular.y = total_action[1]
        training_msg.header.stamp = rospy.Time.now()
        # Setting frame to base_link for planning
        training_msg.header.frame_id = "tray"
        # Need to transform to base_link frame for planning
        rot_vector = geometry_msgs.msg.Vector3Stamped()
        rot_vector.vector = training_msg.twist.angular
        rot_vector.header.frame_id = training_msg.header.frame_id
        try:
            rot_vector = listener.transformVector3("base_link", rot_vector)
        except tf.TransformException:
            ROS_ERROR("%s",ex.what())
            rospy.sleep(1)
            continue
        training_msg.twist.angular = rot_vector.vector
        if training_msg.twist.angular.x > 1:
            training_msg.twist.angular.x = 1
        elif training_msg.twist.angular.x < -1:
            training_msg.twist.angular.x = -1
        if training_msg.twist.angular.y > 1:
            training_msg.twist.angular.y = 1
        elif training_msg.twist.angular.y < -1:
            training_msg.twist.angular.y = -1
        training_msg.header.frame_id = "base_link"

        # Publish action
        T1 = rospy.get_rostime()
        cmd.publish(training_msg)
        rospy.sleep(0.05)

      # Starting updates of weights
      if step > UPDATE_START and step % UPDATE_INTERVAL == 0:
          if first_update:
              print("\nStarting updates")
              first_update = False
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

      # Check if action has been executed for enough time
      T2 = rospy.get_rostime()
      print((T2.nsecs-T1.nsecs)*10e-7)
      while (T2-T1) < rospy.Duration.from_sec(0.2):
          try:
              T2 = rospy.get_rostime()
              continue
          except KeyboardInterrupt:
              break
      # Action executed now calculate reward
      rospy.Subscriber("observations", observations, observation.get_state)
      next_state = torch.tensor(observation.state).to(device)
      # Distance to target:
      d_target = sqrt((goal_x-observation.ball_x)**2+(goal_y-observation.ball_y)**2)
      if reward_dense:
          if d_target < 0.01:
              reward = 10
          else:
              reward = -(d_target)
      elif reward_sparse:
          if d_target < 0.01:
              reward = 10
          else:
              reward = -1
      # Check if Done
      if (d_target<0.01 or step % 200 == 0):
          done = True
          if d_target<0.01:
              print("\nReached target")
              targets_reached += 1
              if step < UPDATE_START:
                  targets_reached_first500 += 1

      # Store (s, a, r, s', d) in replay buffer D
      D.append({'state': state.unsqueeze(0), 'action': action, 'reward': torch.tensor([reward], dtype=torch.float32, device=device), 'next_state': next_state.unsqueeze(0), 'done': torch.tensor([done], dtype=torch.float32, device=device)})
      #T_append = rospy.get_rostime()
      #print("Append", (T_append.nsecs - T_check_done.nsecs)*10e-7)
      # If s' is terminal, reset environment state
      if done and step % SAVE_INTERVAL != 0:
          print("\nResetting")
          if reset_number == 0:
              reset(start_rpy1, move_group, cmd, reset_msg)
              reset_number += 1
          elif reset_number == 1:
              reset(start_rpy2, move_group, cmd, reset_msg)
              reset_number += 1
          else:
              reset(start_rpy3, move_group, cmd, reset_msg)
              reset_number = 0
          done = False

      # Saving policy
      if step % SAVE_INTERVAL == 0:
          # Reset
          print("Saving")
          if reset_number == 0:
              reset(start_rpy1, move_group, cmd, reset_msg)
              reset_number += 1
          elif reset_number == 1:
              reset(start_rpy2, move_group, cmd, reset_msg)
              reset_number += 1
          else:
              reset(start_rpy3, move_group, cmd, reset_msg)
              reset_number = 0
          done = False
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
          },package_path+"/scripts/checkpoints_human/agent.pth")
          print("Saving replay buffer")
          pickle.dump( D, open( package_path+"/scripts/checkpoints_human/agent.p", "wb" ) )

      torch.cuda.empty_cache()
      #train_rate.sleep()

    except KeyboardInterrupt:
        break

print("Finished training")
print("Targets reached with random policy:", targets_reached_first500)
print("Targets reached with overall:", targets_reached)
