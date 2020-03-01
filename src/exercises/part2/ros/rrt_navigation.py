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
from hyperparams_ur10 import OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, UPDATE_INTERVAL, UPDATE_START, SAVE_INTERVAL
from models_ur10_agent import Critic, SoftActor, create_target_network, update_target_network
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

# Do stuff here:
def get_velocity(position, path_points):
  v = np.zeros_like(position)
  if len(path_points) == 0:
    return v
  # Stop moving if the goal is reached.
  if np.linalg.norm(position - path_points[-1]) < .2:
    return v
  # Create an array where every element is the euclidean distance from the current position to points on the path
  distances = []
  for i, element in enumerate(path_points):
      distances.append(np.linalg.norm(element-position))
  # Shortest distance
  closest_point = np.argmin(distances)
  # Set veloccity towards next point in path_points
  if len(path_points) > closest_point+1:
      v = path_points[closest_point+1] - position
  else:
      v = path_points[closest_point] - position
  # Scale v
  v = 5*v
  return v

# Leave as is
class SLAM(object):
  def __init__(self):
    rospy.Subscriber('/map', OccupancyGrid, self.callback)
    self._tf = TransformListener()
    self._occupancy_grid = None
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

  def callback(self, msg):
    values = np.array(msg.data, dtype=np.int8).reshape((msg.info.width, msg.info.height))
    processed = np.empty_like(values)
    processed[:] = rrt.FREE
    processed[values < 0] = rrt.UNKNOWN
    processed[values > 50] = rrt.OCCUPIED
    processed = processed.T
    origin = [msg.info.origin.position.x, msg.info.origin.position.y, 0.]
    resolution = msg.info.resolution
    self._occupancy_grid = rrt.OccupancyGrid(processed, origin, resolution)

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
    # We choose goal:
    self.xpos = 1.5
    self.ypos = 1.5
    self._position = np.array([self.xpos, self.ypos], dtype=np.float32)

  def set_goal_pos(self, goal_pos):
      self.xpos = goal_pos[0]
      self.ypos = goal_pos[1]
      self._position = np.array([self.xpos, self.ypos], dtype=np.float32)

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
    center, radius = rrt.find_circle(u, v)
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
  rospy.init_node('rrt_navigation')

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
  pbar = tqdm(xrange(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
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

    # Follow path using RL agent:
    # Get state:
    position = np.array([
        slam.pose[X] + EPSILON * np.cos(slam.pose[YAW]),
        slam.pose[Y] + EPSILON * np.sin(slam.pose[YAW])], dtype=np.float32)
    # Get action:
    v = get_velocity(position, np.array(current_path, dtype=np.float32))
    u, w = feedback_linearized(slam.pose, v, epsilon=EPSILON)
    # Execute action:
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)

    # Should this still be here?
    # Update plan every 1s.
    time_since = current_time - previous_time
    if current_path and time_since < 2.:
      rate_limiter.sleep()
      continue
    previous_time = current_time

    # Run RRT too find path:
    start_node, final_node = rrt.rrt(slam.pose, goal.position, slam.occupancy_grid)
    current_path = get_path(final_node)
    if not current_path:
      print('Unable to reach goal position:', goal.position)

    # Publish path to RViz.
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

    rate_limiter.sleep()
    frame_id += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs RL RRT navigation')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
