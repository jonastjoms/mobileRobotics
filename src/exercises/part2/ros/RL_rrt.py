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
from sensor_msgs.msg import LaserScan
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
  import rrt_improved as rrt
except ImportError:
  raise ImportError('Unable to import potential_field.py. Make sure this file is in "{}"'.format(directory))

SPEED = .2
EPSILON = .1
X = 0
Y = 1
YAW = 2
resuming = True
testing = False

class Observation:

    def __init__(self):
        self.direction_subgoal_x = 0
        self.direction_subgoal_y = 0
        #self.robot_rotation_x = 0
        #self.robot_rotation_y = 0
        #self.state = np.array([self.direction_subgoal_x, self.direction_subgoal_y, self.robot_rotation_x, self.robot_rotation_y], dtype=float)
        self.state = np.array([self.direction_subgoal_x, self.direction_subgoal_y], dtype=float)

# Method to get state from observations
    def get_state(self, position, subgoal):
        direction_to_subgoal = subgoal.position-position
        self.direction_subgoal_x = direction_to_subgoal[X]
        self.direction_subgoal_y = direction_to_subgoal[Y]
        #self.robot_rotation_x = np.cos(pose[2])
        #self.robot_rotation_y = np.sin(pose[2])
        # Set state
        #self.state = np.array([self.direction_subgoal_x, self.direction_subgoal_y, self.robot_rotation_x, self.robot_rotation_y], dtype=float)
        self.state = np.array([self.direction_subgoal_x, self.direction_subgoal_y], dtype=float)

class SimpleLaser(object):
  def __init__(self):
    rospy.Subscriber('/scan', LaserScan, self.callback)
    self._angles = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
    self._width = np.pi / 180. * 10.  # 10 degrees cone of view.
    self._measurements = [float('inf')] * len(self._angles)
    self._indices = None

  def callback(self, msg):
    # Helper for angles.
    def _within(x, a, b):
      pi2 = np.pi * 2.
      x %= pi2
      a %= pi2
      b %= pi2
      if a < b:
        return a <= x and x <= b
      return a <= x or x <= b;

    # Compute indices the first time.
    if self._indices is None:
      self._indices = [[] for _ in range(len(self._angles))]
      for i, d in enumerate(msg.ranges):
        angle = msg.angle_min + i * msg.angle_increment
        for j, center_angle in enumerate(self._angles):
          if _within(angle, center_angle - self._width / 2., center_angle + self._width / 2.):
            self._indices[j].append(i)

    ranges = np.array(msg.ranges)
    for i, idx in enumerate(self._indices):
      # We do not take the minimum range of the cone but the 10-th percentile for robustness.
      self._measurements[i] = np.percentile(ranges[idx], 10)

  @property
  def ready(self):
    return not np.isnan(self._measurements[0])

  @property
  def measurements(self):
    return self._measurements

def braitenberg(front, front_left, front_right, left, right):

  mask = np.array((1, 0, 0, 0, 0))
  distances = np.array([4 if el == np.inf else el for el in np.array((front, front_left, front_right, left, right))])
  u = 0.3*np.tanh(front-0.3)
  w = np.sum([1.3*distances[1],-1.3*distances[2], distances[3], -distances[4], 0.5*distances[0]]) # [rad/s] going counter-clockwise.

  return u, w

def reset(laser, stop_msg, publisher, slam, goal, path_publisher, frame_id):
    publisher.publish(stop_msg)
    slam.update()
    while not goal.ready or not slam.ready:
      slam.update()
      rate_limiter.sleep()
      continue
    counter = 0
    # Use braitenberg to make sure robot is free from obstacles
    while counter < 100:
        # Make sure all measurements are ready.
        if not laser.ready or not slam.ready:
          rospy.Rate(200).sleep()
          continue
        slam.update()
        u, w = braitenberg(*laser.measurements)
        vel_msg = Twist()
        vel_msg.linear.x = u
        vel_msg.angular.z = w
        publisher.publish(vel_msg)
        counter += 1
    publisher.publish(stop_msg)
    #Find new path
    slam.update()
    start_node, final_node = rrt.rrt(slam.pose, goal.position, slam.occupancy_grid)
    current_path = get_path(final_node)
    #Run RRT too find path:
    while not current_path:
        slam.update()
        start_node, final_node = rrt.rrt(slam.pose, goal.position, slam.occupancy_grid)
        current_path = get_path(final_node)
        if not current_path:
          print('Unable to reach goal position:', goal.position)
          counter = 0
          # Use braitenberg to make sure robot is free from obstacles
          while counter < 100:
              vel_msg = Twist()
              vel_msg.linear.x = -1
              vel_msg.angular.z = 0
              publisher.publish(vel_msg)
              counter += 1
          continue
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
    return current_path

def feedback_linearized(pose, velocity, epsilon):
  u = velocity[X]*np.cos(pose[YAW]) + velocity[Y]*np.sin(pose[YAW])  # [m/s]
  w = (1/epsilon)*(-velocity[X]*np.sin(pose[YAW]) + velocity[Y]*np.cos(pose[YAW]))  # [rad/s] going counter-clockwise.
  return u, w

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
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.callback)
    self._position = np.array([np.nan, np.nan], dtype=np.float32)

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

class SubGoal():
    def __init__(self):
      self.position = np.array([np.nan, np.nan], dtype=np.float32)
      self.path_pointer = 0

    def set_position(self, position):
        self.position[X] = position[X]
        self.position[Y] = position[Y]

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
    rospy.init_node('RL_rrt_training')

    # Torch initialisations
    obs = Observation()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    state = torch.tensor(obs.state, dtype=torch.float32).to(device)
    done = False
    # Update this between trials
    trial = 0
    episode = 0
    episode_reward = 0
    episode_error = 0
    rewards = {}
    errors = {}
    reward_sum = 0
    episode_step = 1
    first_update = True
    positions = []

    # SAC initialisations
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
        UPDATE_START = 0

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
        #rewards = pickle.load( open("/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/training_rewards.p", "rb" ) )
        #errors = pickle.load( open("/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/errors.p", "rb" ) )

    # Other variables
    reward_sparse = True
    reward_dense = False
    pbar = tqdm(xrange(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)

    # Update control every 200 ms.
    rate_limiter = rospy.Rate(200)
    publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
    path_publisher = rospy.Publisher('/path', Path, queue_size=1)
    slam = SLAM()
    goal = GoalPose()
    subgoal = SubGoal()
    laser = SimpleLaser()
    frame_id = 0
    current_path = []
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

    # Make sure all measurements are ready.
    # Get map and current position through SLAM:
    # > roslaunch exercises slam.launch
    while not goal.ready or not slam.ready:
        slam.update()
        rate_limiter.sleep()
        continue

    # Find initial path
    # Run RRT too find path:
    while not current_path:
        start_node, final_node = rrt.rrt(slam.pose, goal.position, slam.occupancy_grid)
        current_path = get_path(final_node)
        if not current_path:
          print('Unable to reach goal position:', goal.position)
          current_path = reset(laser, stop_msg, publisher, slam, goal, path_publisher, frame_id)
          continue
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

    # Set initial subgoal position, 4 path points away, because of feedback linearization
    subgoal.path_pointer = 4
    subgoal.set_position(np.array(current_path[subgoal.path_pointer]))


    # Training loop
    for step in pbar:
        slam.update()
        # Make sure all measurements are ready.
        # Get map and current position through SLAM:
        # > roslaunch exercises slam.launch
        while not goal.ready or not slam.ready:
            slam.update()
            rate_limiter.sleep()
        # RL:
        try:
            with torch.no_grad():
                # Get state:
                position = np.array([
                    slam.pose[X] + EPSILON * np.cos(slam.pose[YAW]),
                    slam.pose[Y] + EPSILON * np.sin(slam.pose[YAW])], dtype=np.float32)
                obs.get_state(position, subgoal)
                state = torch.tensor(obs.state).float().to(device)
                # Determine action:
                if step < UPDATE_START:
                    # To improve exploration take actions sampled from a uniform random distribution over actions at the start of training
                    action = torch.tensor([2 * random.random() - 1, 2 * random.random() - 1], device=device).unsqueeze(0)
                else:
                    # Observe state s and select action a ~ mu(a|s)
                    action = actor(state.unsqueeze(0)).sample()
                # Scale action so that we don't reach max = 0.5
                action = action/4
                # Get forward and rotational velocity:
                u, w = feedback_linearized(slam.pose, action.numpy()[0], epsilon=EPSILON)
                #u = action.numpy()[0][0]
                #w = action.numpy()[0][1]/2
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
            while (T2-T1) < rospy.Duration.from_sec(0.3):
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
            obs.get_state(position, subgoal)
            next_state = torch.tensor(obs.state).float().to(device)
            # Distance to end goal:
            goal_reached = np.linalg.norm(goal.position-position) < .2
            # Check if reached final goal or final step in episode
            if (goal_reached or step % 200 == 0):
                done = True
                if goal_reached:
                    reward = 1
                else:
                    reward = -1
                episode_reward += reward
                if not testing:
                    key = episode
                    rewards[key] = episode_reward/episode_step
                    errors[key] = episode_error/episode_step
                    episode_reward = 0
                    episode_error = 0
                    episode_step = 0
                    print(positions)
                    print(current_path)
                    positions = []
                episode += 1
                # Make sure the robot is stopped.
                i = 0
                while i < 10 and not rospy.is_shutdown():
                  publisher.publish(stop_msg)
                  rate_limiter.sleep()
                  i += 1
                if testing:
                    rospy.signal_shutdown("Done")
                    break
                current_path = reset(laser, stop_msg, publisher, slam, goal, path_publisher, frame_id)
                # Update subgoal
                subgoal.path_pointer = 4
                subgoal.set_position(np.array(current_path[subgoal.path_pointer]))

            # Check whether subgoal reached and must be updated
            sub_goal_reached = np.linalg.norm(subgoal.position-position) < 0.08
            if (sub_goal_reached and step % 200 != 0):
                done = True
                print("")
                print("Goal reached at: ", state)
                reward = 5
                # Update subgoal
                subgoal.path_pointer += 2
                # Handle end of path:
                if subgoal.path_pointer > len(current_path) - 1:
                    subgoal.path_pointer = len(current_path) - 1
                # Set new subgoal
                subgoal.set_position(np.array(current_path[subgoal.path_pointer]))
            else:
                reward = -1

            # Calculate error as distance to subgoal closest point on path
            error = np.linalg.norm(subgoal.position-position)

            episode_reward += reward
            episode_error += error
            episode_step += 1
            p = (position[0], position[1])
            positions.append(p)

            # Store (s, a, r, s', d) in replay buffer D
            #print("")
            #print("state", state.numpy())
            #print("Pos", position)
            #print("SubGoal", subgoal.position)
            #print("action", action.numpy())
            #print("reward", reward)
            #print("next state", next_state.numpy())
            D.append({'state': state.unsqueeze(0), 'action': action, 'reward': torch.tensor([reward], dtype=torch.float32, device=device), 'next_state': next_state.unsqueeze(0), 'done': torch.tensor([done], dtype=torch.float32, device=device)})
            done = False
            # Saving policy
            if step % SAVE_INTERVAL == 0:
                current_path = reset(laser, stop_msg, publisher, slam, goal, path_publisher, frame_id)
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
                },"/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/agent2.pth")
                print("Saving replay buffer")
                pickle.dump( D, open( "/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/deque2.p", "wb" ) )
                pickle.dump( rewards, open( "/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/training_rewards2.p", "wb" ) )
                pickle.dump( errors, open( "/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/errors2.p", "wb" ) )

            torch.cuda.empty_cache()
            frame_id += 1

        except KeyboardInterrupt:
            break
        except rospy.ROSTimeMovedBackwardsException:
            continue


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs RL RRT navigation')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
