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
from hrc_msgs.msg import observations_human_human
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
        self.human1_theta = 0
        self.human1_phi = 0
        self.human2_theta = 0
        self.human2_phi = 0
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
        self.human1_theta = obs.human1_theta
        self.human1_phi = obs.human1_phi
        self.human2_theta = obs.human2_theta
        self.human2_phi = obs.human2_phi
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

rospy.Subscriber("observations_human_human", observations_human_human, observation.get_state)
state = torch.tensor(observation.state)
done = False
reward = 0
done = False
reward = 0

# Launching jogger
uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)
launch = roslaunch.parent.ROSLaunchParent(uuid, [package_path+"/launch/jogger.launch"])
launch.start()
rospy.sleep(3)

# Other variables
reward_sparse = True
reward_dense = False
goal_x = 0.17
goal_y = -0.15
abs_max_theta = 0.1
abs_max_phi = 0.1
human_p = 5

# Reset
print("Testing policy")
done = False
reward = 0
# Test

# Testrun
i = 1 # Steps in testing run
total_reward = 0
print("Testing started")
while not done:
    # Get observation
    rospy.Subscriber("observations_human_human", observations_human_human, observation.get_state)
    # Must compensate for translation so get current pose
    current_pose = move_group.get_current_pose().pose
    current_rpy = move_group.get_current_rpy()
    # Scaling human actions
    # Human 1
    human_cmd1 = human_p * (observation.human1_theta - observation.robot_theta)
    if human_cmd1 >= 1:
        human_cmd1 = 0.99
    elif human_cmd1 <= -1:
        human_cmd1 = -0.99
    # Human 2:
    human_cmd2 = human_p * (observation.human2_phi - observation.robot_phi)
    if human_cmd2 >= 1:
        human_cmd2 = 0.99
    elif human_cmd2 <= -1:
        human_cmd2 = -0.99
    # Defining total action
    total_action = [human_cmd1, human_cmd2]
    # print(current_rpy)
    # Setting linear commands to be published
    training_msg.twist.linear.x = pose_goal.position.x - current_pose.position.x
    training_msg.twist.linear.y = pose_goal.position.y - current_pose.position.y
    training_msg.twist.linear.z = pose_goal.position.z - current_pose.position.z
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
    training_msg.twist.angular.z = 0
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
    cmd.publish(training_msg)
    rospy.sleep(0.2)

    # Action executed now calculate reward
    rospy.Subscriber("observations_human_human", observations_human_human, observation.get_state)
    next_state = torch.tensor(observation.state)
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
    if (d_target < 0.01 or i % 200 == 0):
      done = True
      break
    total_reward += reward
    i +=1
training_msg.twist.linear.x = 0
training_msg.twist.linear.y = 0
training_msg.twist.linear.z = 0
training_msg.twist.angular.x = 0
training_msg.twist.angular.y = 0
training_msg.twist.angular.z = 0
print("Testing finished, total reward:")
print(total_reward)
