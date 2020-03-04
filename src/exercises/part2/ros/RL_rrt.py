#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import argparse
import numpy as np
import os
import rospy
import sys
import scipy
import copy
from math import pi, sqrt
from collections import deque
import random
import torch
import time
import cPickle as pickle
from torch import optim
from tqdm import tqdm
from hyperparams import OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, UPDATE_INTERVAL, UPDATE_START, SAVE_INTERVAL
from models import Critic, SoftActor, create_target_network, update_target_network
from decimal import Decimal

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Occupancy grid.
from nav_msgs.msg import OccupancyGrid
# Position.
from tf import TransformListener
# Goal.
from geometry_msgs.msg import PoseStamped
# Path.
from nav_msgs.msg import Path
# For pose information.
from tf.transformations import euler_from_quaternion
from std_srvs.srv import Empty


# Import the rrt code rather than copy-pasting.
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python')
sys.path.insert(0, directory)
try:
  import rrt_improved
except ImportError:
  raise ImportError('Unable to import potential_field.py. Make sure this file is in "{}"'.format(directory))

SPEED = .2
EPSILON = .1
X = 0
Y = 1
YAW = 2
resuming = True
# Reset simulation service:
service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

class Observation:

    def __init__(self):
        self.X = 0
        self.Y = 0
        self.closest_point_on_path = [0,0]
        self.next_point_on_path = [0,0]
        self.state = np.array([self.X, self.Y, self.closest_point_on_path[0], self.closest_point_on_path[1], self.next_point_on_path[0], self.next_point_on_path[1]], dtype=float)
# Method to get state from observations
    def get_state(self, position, path_points):
        self.X = position[X]
        self.Y = position[Y]
        # Create an array where every element is the euclidean distance from the current position to points on the path
        distances = []
        for i, element in enumerate(path_points):
            distances.append(np.linalg.norm(element-position))
        # Shortest distance to that point
        closest_point = np.argmin(distances)
        self.closest_point_on_path = np.array(path_points)[closest_point]
        if len(path_points) > closest_point + 1:
            self.next_point_on_path = np.array(path_points)[closest_point+1]
        else:
            self.next_point_on_path = self.closest_point_on_path
        self.state = np.array([self.X, self.Y, self.closest_point_on_path[0], self.closest_point_on_path[1], self.next_point_on_path[0], self.next_point_on_path[1]], dtype=float)

def reset(slam):
    response = service()
    return

def new_path(slam, goal, frame_id, path_publisher):
    # Run RRT too find path:
    start_node, final_node = rrt_improved.rrt(slam.pose, goal.position, slam.occupancy_grid)
    current_path = get_path(final_node)
    if not current_path:
      print('Unable to reach goal position:', goal.position)
    # Publish path
    path_msg = Path()
    path_msg.header.seq = frame_id
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = 'map'
    for u in current_path:
      pose_msg = PoseStamped()
      pose_msg.header.seq = frame_id
      pose_msg.header.stamp = path_msg.header.stamp
      pose_msg.header.frame_id = 'map'
      pose_msg.pose.position.x = u[X]
      pose_msg.pose.position.y = u[Y]
      path_msg.poses.append(pose_msg)
    path_publisher.publish(path_msg)
    return current_path, path_msg

def feedback_linearized(pose, velocity, epsilon):
  # Get theta_dot
  theta_dot = np.arctan2(velocity[Y], velocity[X])-pose[YAW]
  # Arctan2 gives angles from -pi to pi, must handle
  theta_dot = np.mod(theta_dot + np.pi, 2*np.pi)-np.pi
  # Feedback linearization:
  x_p_dot = velocity[X] + epsilon*(-theta_dot*np.sin(pose[YAW]))
  y_p_dot = velocity[Y] + epsilon*(theta_dot*np.cos(pose[YAW]))
  u = x_p_dot*np.cos(pose[YAW]) + y_p_dot*np.sin(pose[YAW])  # [m/s]
  w = (1/epsilon)*(-x_p_dot*np.sin(pose[YAW])+ y_p_dot*np.cos(pose[YAW]))  # [rad/s] going counter-clockwise.

  return u, w

# Leave as is
class SLAM(object):
  def __init__(self):
    rospy.Subscriber('/map', OccupancyGrid, self.callback)
    self._tf = TransformListener()
    self._occupancy_grid = None
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float)

  def callback(self, msg):
    values = np.array(msg.data, dtype=np.int8).reshape((msg.info.width, msg.info.height))
    processed = np.empty_like(values)
    processed[:] = rrt_improved.FREE
    processed[values < 0] = rrt_improved.UNKNOWN
    processed[values > 50] = rrt_improved.OCCUPIED
    processed = processed.T
    origin = [msg.info.origin.position.x, msg.info.origin.position.y, 0.]
    resolution = msg.info.resolution
    self._occupancy_grid = rrt_improved.OccupancyGrid(processed, origin, resolution)

  def update(self):
    # Get pose w.r.t. map.
    a = 'occupancy_grid'
    b = 'base_link'
    if self._tf.frameExists(a) and self._tf.frameExists(b):
      try:
        t = rospy.Time(0)
        position, orientation = self._tf.lookupTransform('/' + a, '/' + b, t)
        self._pose[X] = position[X]
        self._pose[Y] = position[Y]
        _, _, self._pose[YAW] = euler_from_quaternion(orientation)
      except Exception as e:
        print(e)
    else:
      print('Unable to find:', self._tf.frameExists(a), self._tf.frameExists(b))
    pass

  @property
  def ready(self):
    return self._occupancy_grid is not None and not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose

  @property
  def occupancy_grid(self):
    return self._occupancy_grid

class GoalPose(object):
  def __init__(self):
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.callback)
    self._position = np.array([np.nan, np.nan], dtype=np.float)

  def callback(self, msg):
    # The pose from RViz is with respect to the "map".
    self._position[X] = msg.pose.position.x
    self._position[Y] = msg.pose.position.y
    print('Received new goal position:', self._position)

  @property
  def ready(self):
    return not np.isnan(self._position[0])

  @property
  def position(self):
    return self._position

# Leave as is
def get_path(final_node):
  # Construct path from RRT solution.
  if final_node is None:
    return []
  path_reversed = []
  path_reversed.append(final_node)
  while path_reversed[-1].parent is not None:
    path_reversed.append(path_reversed[-1].parent)
  path = list(reversed(path_reversed))
  # Put a point every 5 cm.
  distance = 0.05
  offset = 0.
  points_x = []
  points_y = []
  for u, v in zip(path, path[1:]):
    center, radius = rrt_improved.find_circle(u, v)
    du = u.position - center
    theta1 = np.arctan2(du[1], du[0])
    dv = v.position - center
    theta2 = np.arctan2(dv[1], dv[0])
    # Check if the arc goes clockwise.
    clockwise = np.cross(u.direction, du).item() > 0.
    # Generate a point every 5cm apart.
    da = distance / radius
    offset_a = offset / radius
    if clockwise:
      da = -da
      offset_a = -offset_a
      if theta2 > theta1:
        theta2 -= 2. * np.pi
    else:
      if theta2 < theta1:
        theta2 += 2. * np.pi
    angles = np.arange(theta1 + offset_a, theta2, da)
    offset = distance - (theta2 - angles[-1]) * radius
    points_x.extend(center[X] + np.cos(angles) * radius)
    points_y.extend(center[Y] + np.sin(angles) * radius)
  return zip(points_x, points_y)

# Here the fun part goes:
def run(args):
  rospy.init_node('RL_rrt')

  # Torch initialisations
  obs = Observation()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
  state = torch.tensor(obs.state, dtype=torch.float32).to(device)
  done = False
  rewards = []
  reward_sum = 0
  first_update = True


  # SAC initialisations
  rewards = []
  action_space = 2
  state_space = 3
  actor = SoftActor(HIDDEN_SIZE).to(device)
  critic_1 = Critic(HIDDEN_SIZE, state_action=True).to(device)
  critic_2 = Critic(HIDDEN_SIZE, state_action=True).to(device)
  value_critic = Critic(HIDDEN_SIZE).to(device)
  if resuming:
      print("Loading models")
      checkpoint = torch.load("/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/agent.pth")
      actor.load_state_dict(checkpoint['actor_state_dict'])
      critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
      critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
      value_critic.load_state_dict(checkpoint['value_critic_state_dict'])
      UPDATE_START = 1

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
  if resuming:
      target_value_critic.load_state_dict(checkpoint['target_value_critic_state_dict'])
      actor_optimiser.load_state_dict(checkpoint['actor_optimiser_state_dict'])
      critics_optimiser.load_state_dict(checkpoint['critics_optimiser_state_dict'])
      value_critic_optimiser.load_state_dict(checkpoint['value_critic_optimiser_state_dict'])
      alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
      D = pickle.load( open("/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/deque.p", "rb" ) )

  # Other variables
  reward_sparse = False
  reward_dense = True
  pbar = tqdm(xrange(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)

  # Update control every 200 ms.
  rate_limiter = rospy.Rate(200)
  publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
  path_publisher = rospy.Publisher('/path', Path, queue_size=1)
  slam = SLAM()
  goal = GoalPose()
  frame_id = 0
  current_path = []
  previous_time = rospy.Time.now().to_sec()
  # Stop moving message.
  stop_msg = Twist()
  stop_msg.linear.x = 0.
  stop_msg.angular.z = 0.

  # Make sure the robot is stopped.
  i = 0
  while i < 10 and not rospy.is_shutdown():
    publisher.publish(stop_msg)
    rate_limiter.sleep()
    i += 1
  while not rospy.is_shutdown():
    slam.update()
    current_time = rospy.Time.now().to_sec()

    # Make sure all measurements are ready.
    # Get map and current position through SLAM:
    # > roslaunch exercises slam.launch
    if not goal.ready or not slam.ready:
      rate_limiter.sleep()
      continue
    goal_reached = np.linalg.norm(slam.pose[:2] - goal.position) < .2
    if goal_reached:
      publisher.publish(stop_msg)
      rate_limiter.sleep()
      continue

    # Reset and get observation:
    reset(slam)
    current_path, path_msg = new_path(slam, goal, frame_id, path_publisher)
    current_path = []
    while not current_path:
        current_path, path_msg = new_path(slam, goal, frame_id, path_publisher)
    #pbar = range(500)
    #pbar.pop(0)
    # Training loop
    for step in pbar:
        slam.update()
        try:
            with torch.no_grad():
                path_publisher.publish(path_msg)
                position = np.array([
                # Get state:
                    slam.pose[X] + EPSILON * np.cos(slam.pose[YAW]),
                    slam.pose[Y] + EPSILON * np.sin(slam.pose[YAW])], dtype=np.float32)
                obs.get_state(position, current_path)
                state = torch.tensor(obs.state).float().to(device)
                current_time = rospy.Time.now().to_sec()
                # Determine action:
                if step < UPDATE_START:
                  # To improve exploration take actions sampled from a uniform random distribution over actions at the start of training
                  action = torch.tensor([2 * random.random() - 1, 2 * random.random() - 1], device=device).unsqueeze(0)
                else:
                  # Observe state s and select action a ~ mu(a|s)
                  action = actor(state.unsqueeze(0)).sample()
                # Scale action so that we don't reach max = 0.5
                action = action/2
                # Get forward and rotational velocity:
                #u, w = feedback_linearized(slam.pose, action.numpy()[0], epsilon=EPSILON)
                u = action.numpy()[0][0]
                w = action.numpy()[0][1]
                # Execute action:
                vel_msg = Twist()
                vel_msg.linear.x = u
                vel_msg.angular.z = w
                publisher.publish(vel_msg)

                # Get current time and set delay
                T1 = rospy.get_rostime()

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

            # Check if action has been executed long enough
            T2 = rospy.get_rostime()
            while (T2-T1) < rospy.Duration.from_sec(0.2):
                try:
                    T2 = rospy.get_rostime()
                    continue
                except KeyboardInterrupt:
                    break

            # Action executed now calculate reward
            slam.update()
            position = np.array([
                slam.pose[X] + EPSILON * np.cos(slam.pose[YAW]),
                slam.pose[Y] + EPSILON * np.sin(slam.pose[YAW])], dtype=np.float32)
            obs.get_state(position, current_path)
            next_state = torch.tensor(obs.state).float().to(device)
            # Reward, two parts: (Distance to next point on path)
            reward = - 3*np.linalg.norm(obs.next_point_on_path-position)
            #print("distance reward",reward)
            # (Difference in path direction, and robot direction, only included when almost on path
            robot_direction = np.array([np.cos(slam.pose[YAW]), np.sin(slam.pose[YAW])], dtype=np.float32)
            if np.linalg.norm(robot_direction) > 0:
                robot_direction = robot_direction/np.linalg.norm(robot_direction)
            path_direction = obs.next_point_on_path - obs.closest_point_on_path
            if np.linalg.norm(path_direction) > 0:
                path_direction = path_direction/np.linalg.norm(path_direction)
            # Get angle between:
            angle_between = np.arccos(np.clip(np.dot(robot_direction,path_direction), -1.0,1.0))
            #print("angle reward",-angle_between/5)
            if np.abs(reward/3) < 0.1:
                reward -= angle_between/5
            reward_sum += reward

            # Distance to goal:
            goal_reached = np.linalg.norm(slam.pose[:2] - goal.position) < .3
            # Check if Done
            if (goal_reached or step % 200 == 0):
                done = True
                rewards.append(reward_sum)
                reward_sum = 0
                publisher.publish(stop_msg)
                reset(slam)
                rate_limiter.sleep()

            # Store (s, a, r, s', d) in replay buffer D
            #print("state", state.numpy())
            #print("action", action.numpy())
            #print("reward", reward)
            #print("next state", next_state.numpy())
            #publisher.publish(stop_msg)
            D.append({'state': state.unsqueeze(0), 'action': action, 'reward': torch.tensor([reward], dtype=torch.float32, device=device), 'next_state': next_state.unsqueeze(0), 'done': torch.tensor([done], dtype=torch.float32, device=device)})
            state = next_state
            done = False
            # Saving policy
            if step % SAVE_INTERVAL == 0:
                print(rewards[-1])
                reset(slam)
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
                },"/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/agent.pth")
                print("Saving replay buffer")
                pickle.dump( D, open( "/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/deque.p", "wb" ) )

            # Update plan every 600 step.
            if step % 100 == 0:
                publisher.publish(stop_msg)
                reset(slam)
                rate_limiter.sleep()
                slam.update()
                rate_limiter.sleep()
                current_path, path_msg = new_path(slam, goal, frame_id, path_publisher)
                while not current_path:
                    rate_limiter.sleep()
                    current_path, path_msg = new_path(slam, goal, frame_id, path_publisher)

            torch.cuda.empty_cache()

            frame_id += 1

        except KeyboardInterrupt:
            break


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs RL RRT navigation')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
