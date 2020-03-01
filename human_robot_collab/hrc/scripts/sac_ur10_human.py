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
from hyperparams_ur10 import OFF_POLICY_BATCH_SIZE as BATCH_SIZE, DISCOUNT, ENTROPY_WEIGHT, HIDDEN_SIZE, LEARNING_RATE, MAX_STEPS, POLYAK_FACTOR, REPLAY_SIZE, UPDATE_INTERVAL, UPDATE_START, SAVE_INTERVAL
from models_ur10_human import Critic, SoftActor, create_target_network, update_target_network
from decimal import Decimal
import rospkg

rospack = rospkg.RosPack()
package_path = rospack.get_path("hrc")

# Observation class
class RL:

    def __init__(self):
        self.robot_theta = 0
        self.robot_phi = 0
        self.robot_theta_dot = 0
        self.robot_phi_dot = 0
        self.ball_x = 0
        self.ball_y = 0
        self.ball_vel_x = 0
        self.ball_vel_y = 0
        self.human_theta = 0
        self.human_phi = 0
        self.state = [self.robot_theta, self.robot_phi, self.robot_theta_dot,
        self.robot_phi_dot, self.ball_x, self.ball_y,
        self.ball_vel_x, self.ball_vel_y]

# Method to get state from observations
    def get_state(self, obs):
        self.robot_theta = obs.robot_theta
        self.robot_phi = obs.robot_phi
        self.robot_theta_dot = obs.robot_theta_dot
        self.robot_phi_dot = obs.robot_phi_dot
        self.ball_x = obs.ball_x
        self.ball_y = obs.ball_y
        self.ball_vel_x = obs.ball_vel_x
        self.ball_vel_y = obs.ball_vel_y
        self.human_theta = obs.human_theta
        self.human_phi = obs.human_phi
        self.state = [self.robot_theta, self.robot_phi, self.robot_theta_dot,
        self.robot_phi_dot, self.ball_x, self.ball_y,
        self.ball_vel_x, self.ball_vel_y]

# Method used to check if robot is at reset position:
def all_close(goal, actual, tolerance):
    all_equal = True

    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True

# Method to reset robot
def initialize(move_group, pose_goal):

    move_group.set_pose_target(pose_goal)
    ## Now, we call the planner to compute the plan and execute it.
    plan = move_group.go(wait=True)
    # Calling `stop()` ensures that there is no residual movement
    move_group.stop()
    # Checking that goal is reached:
    current_pose = move_group.get_current_pose().pose
    while not all_close(pose_goal, current_pose, 0.01):
        continue
    return True

def reset(start_rpy, move_group, cmd, reset_msg):
    current_rpy = np.array(move_group.get_current_rpy())
    start_rpy = np.array(start_rpy)
    reset_msg.twist.linear.x = 0
    reset_msg.twist.linear.y = 0
    reset_msg.twist.linear.z = 0
    reset_msg.twist.angular.x = 0
    reset_msg.twist.angular.y = 0
    reset_msg.twist.angular.z = 0
    while not all(abs(i) < 0.005 for i in (start_rpy - current_rpy)):
        try:
            current_rpy = np.array(move_group.get_current_rpy())
            reset_msg.twist.angular.x = 4*(start_rpy[0] - current_rpy[0])
            reset_msg.twist.angular.y = 4*(start_rpy[1] - current_rpy[1])
            reset_msg.twist.angular.z = 4*(start_rpy[2] - current_rpy[2])
            if reset_msg.twist.angular.x > 1:
                reset_msg.twist.angular.x = 1
            elif reset_msg.twist.angular.x < -1:
                reset_msg.twist.angular.x = -1
            if reset_msg.twist.angular.y > 1:
                reset_msg.twist.angular.y = 1
            elif reset_msg.twist.angular.y < -1:
                reset_msg.twist.angular.y = -1
            cmd.publish(reset_msg)
        except KeyboardInterrupt:
            break
    print("Done resetting")
    reset_msg.twist.angular.x = 0
    reset_msg.twist.angular.y = 0
    reset_msg.twist.angular.z = 0
    cmd.publish(reset_msg)
    rospy.sleep(3)
    return True

# Initialize moveit stuff:
## First initialize `moveit_commander`_ and a `rospy`_ node:
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('sac_ur10', anonymous=True)
## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
## kinematic model and the robot's current joint states
robot = moveit_commander.RobotCommander()
## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
## for getting, setting, and updating the robot's internal understanding of the
## surrounding world:
scene = moveit_commander.PlanningSceneInterface()
## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
## to a planning group (group of joints).  In this tutorial the group is the primary
## arm joints in the Panda robot, so we set the group's name to "panda_arm".
## If you are using a different robot, change this value to the name of your robot
## arm planning group.
## This interface can be used to plan and execute motions:
group_name = "manipulator"
move_group = moveit_commander.MoveGroupCommander(group_name)
## Create a `DisplayTrajectory`_ ROS publisher which is used to display
## trajectories in Rviz:
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

# Initialize publisher and listener
cmd = rospy.Publisher('jog_arm_server/delta_jog_cmds', geometry_msgs.msg.TwistStamped, queue_size=100)
listener = tf.TransformListener()
training_msg = geometry_msgs.msg.TwistStamped()
testing_msg = geometry_msgs.msg.TwistStamped()
reset_msg = geometry_msgs.msg.TwistStamped()
observation = RL()

# Define reset/initial pose:
start_rpy1 = [-0.1, -0.1, 0.0]
start_rpy2 = [0.1, -0.1, 0.0]
start_rpy3 = [-0.1, 0.1, 0.0]
start_pose = tf.transformations.quaternion_from_euler(start_rpy1[0], start_rpy1[1], start_rpy1[2] )
pose_goal = geometry_msgs.msg.Pose()
pose_goal.orientation.x = start_pose[0]
pose_goal.orientation.y = start_pose[1]
pose_goal.orientation.z = start_pose[2]
pose_goal.orientation.w = start_pose[3]
pose_goal.position.x = 0.255
pose_goal.position.y = 0.2
pose_goal.position.z = 0.6
initialize(move_group, pose_goal)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
rospy.Subscriber("observations", observations, observation.get_state)
state = torch.tensor(observation.state).to(device)
done = False
reward = 0

# Launching jogger
uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)
launch = roslaunch.parent.ROSLaunchParent(uuid, [package_path+"/launch/jogger.launch"])
launch.start()
rospy.sleep(3)

# SAC initialisations
action_space = 2
state_space = 8
actor = SoftActor(HIDDEN_SIZE).to(device)
critic_1 = Critic(HIDDEN_SIZE, state_action=True).to(device)
critic_2 = Critic(HIDDEN_SIZE, state_action=True).to(device)
value_critic = Critic(HIDDEN_SIZE).to(device)
# If resuming training, load models
if sys.argv[-1] == "resuming":
    if os.path.isfile(package_path+"/scripts/checkpoints_human/agent.pth"):
        print("\nResuming training\n")
        checkpoint = torch.load(package_path+"/scripts/checkpoints_human/agent.pth")
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        value_critic.load_state_dict(checkpoint['value_critic_state_dict'])
        UPDATE_START = 1
        print(UPDATE_START)
    else:
        print("No checkpoint found at '{}'".format(package_path+"/scripts/checkpoints_human/agent.pth"))

target_value_critic = create_target_network(value_critic).to(device)
actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LEARNING_RATE)
value_critic_optimiser = optim.Adam(value_critic.parameters(), lr=LEARNING_RATE)
D = deque(maxlen=REPLAY_SIZE)
# Automatic entropy tuning init
target_entropy = -np.prod(action_space).item()
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optimizer = optim.Adam([log_alpha], lr=LEARNING_RATE)

# If resuming training, load models
if sys.argv[-1] == "resuming":
    if os.path.isfile(package_path+"/scripts/checkpoints_human/agent.pth"):
        target_value_critic.load_state_dict(checkpoint['target_value_critic_state_dict'])
        actor_optimiser.load_state_dict(checkpoint['actor_optimiser_state_dict'])
        critics_optimiser.load_state_dict(checkpoint['critics_optimiser_state_dict'])
        value_critic_optimiser.load_state_dict(checkpoint['value_critic_optimiser_state_dict'])
        alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        D = pickle.load( open( package_path+"/scripts/checkpoints_human/agent.p", "rb" ) )

# Other variables
reward_sparse = True
reward_dense = False
goal_x = 0.17
goal_y = -0.14
abs_max_theta = 0.1
abs_max_phi = 0.1
human_p = 5
pbar = tqdm(xrange(1, MAX_STEPS + 1), unit_scale=1, smoothing=0)
first_update = True
reset_number = 0
targets_reached_first500 = 0
targets_reached = 0


# Training loop
for step in pbar:
    try:
      with torch.no_grad():
        rospy.Subscriber("observations", observations, observation.get_state)
        state = torch.tensor(observation.state).to(device)
        if step < UPDATE_START and sys.argv[-1] != "resuming":
          # To improve exploration take actions sampled from a uniform random distribution over actions at the start of training
          action = torch.tensor([2 * random.random() - 1], device=device).unsqueeze(0)
        else:
          # Observe state s and select action a ~ mu(a|s)
          action = actor(state.unsqueeze(0)).sample()
        # Execute a in the environment and observe next state s', reward r, and done signal d to indicate whether s' is terminal
        # Must compensate for translation
        current_pose = move_group.get_current_pose().pose
        current_rpy = move_group.get_current_rpy()
        # Scaling human action
        human_cmd = human_p * (observation.human_phi - observation.robot_phi)
        if human_cmd >= 1:
            human_cmd = 0.99
        elif human_cmd <= -1:
            human_cmd = -0.99
        # Defining total action
        total_action = [action[0], human_cmd]
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
      d_target = sqrt((goal_x-next_state[4])**2+(goal_y-next_state[5])**2)
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
