from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import numpy as np


WALL_OFFSET = 2.
VIRTUAL_POSITION = np.array([-.5, 1], dtype=np.float32)
VIRTUAL_RADIUS = .1
CYLINDER_POSITION = np.array([.5, .0], dtype=np.float32)
CYLINDER_POSITION = np.array([.3, .2], dtype=np.float32)
CYLINDER_POSITION2 = np.array([.0, .5], dtype=np.float32)
CYLINDER_RADIUS = .3
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)
START_POSITION = np.array([-1.5, -1.5], dtype=np.float32)
MAX_SPEED = .5


def get_velocity_to_reach_goal(position, goal_position):
  # Damping variable:
  zeta = 1
  # Euclidean distance:
  dist = np.linalg.norm(goal_position-position)
  # Determine gradient
  grad = (goal_position-position)/dist
  # Multiply with damping variable
  v = zeta*grad
  # Add stochasticity
  v = [np.random.normal(v[0], 0.1), np.random.normal(v[1], 0.1)]
  if v[0] > 0.5:
      v[0] = 0.5
  if v[1] > 0.5:
      v[1] = 0.5
  return v


def get_velocity_to_avoid_obstacles(position, obstacle_positions, obstacle_radii):
  # Damping variable:
  zeta = 0.5
  # Threshold
  b = 0.8
  # Euclidean distance to center:
  dist = np.linalg.norm(obstacle_positions-position)
  # Determine gradient
  grad = -(obstacle_positions-position)/(np.max((dist - obstacle_radii, b)))**2
  # If too far from obstacle set force to zero:
  if dist > 1:
      zeta = zeta*0
  # Add some rotational force
  grad = grad[0]
  def perpendicular(v):
    w = np.empty_like(v)
    print(v[0])
    w[0] = -v[1]
    w[1] = v[0]
    return w
  perp = perpendicular(grad)
  # Multiply with damping variable
  v = zeta*grad + 0.2*perp
  return v

def get_velocity_to_virtual_fields(position, virtual_position):
  # Damping variable:
  zeta = 0.5
  # Euclidean distance:
  dist = np.linalg.norm(virtual_position-position)
  # Determine gradient
  grad = (virtual_position-position)/dist
  # Multiply with damping variable
  v = zeta*grad
  # Add stochasticity
  v = [np.random.normal(v[0], 0.1), np.random.normal(v[1], 0.1)]
  return v


def normalize(v):
  n = np.linalg.norm(v)
  if n < 1e-2:
    return np.zeros_like(v)
  return v / n


def cap(v, max_speed):
  n = np.linalg.norm(v)
  if n > max_speed:
    return v / n * max_speed
  return v

def get_velocity(position, mode='all'):
  if mode in ('goal', 'all'):
    v_goal = get_velocity_to_reach_goal(position, GOAL_POSITION)
    v_virtual = get_velocity_to_virtual_fields(position, VIRTUAL_POSITION)
  else:
    v_goal = np.zeros(2, dtype=np.float32)
  if mode in ('obstacle', 'all'):
    v_avoid = get_velocity_to_avoid_obstacles(
      position,
      [CYLINDER_POSITION],
      [CYLINDER_RADIUS])
    #v_avoid += get_velocity_to_avoid_obstacles(
      #position,
      #[CYLINDER_POSITION2],
      #[CYLINDER_RADIUS])
    v_virtual = get_velocity_to_virtual_fields(position, VIRTUAL_POSITION)
  else:
    v_avoid = np.zeros(2, dtype=np.float32)
  v = v_goal + v_avoid #+ v_virtual
  return cap(v, max_speed=MAX_SPEED)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs obstacle avoidance with a potential field')
  parser.add_argument('--mode', action='store', default='all', help='Which velocity field to plot.', choices=['obstacle', 'goal', 'all'])
  args, unknown = parser.parse_known_args()

  fig, ax = plt.subplots()
  # Plot field.
  X, Y = np.meshgrid(np.linspace(-WALL_OFFSET, WALL_OFFSET, 30),
                     np.linspace(-WALL_OFFSET, WALL_OFFSET, 30))
  U = np.zeros_like(X)
  V = np.zeros_like(X)
  for i in range(len(X)):
    for j in range(len(X[0])):
      velocity = get_velocity(np.array([X[i, j], Y[i, j]]), args.mode)
      U[i, j] = velocity[0]
      V[i, j] = velocity[1]
  plt.quiver(X, Y, U, V, units='width')

  # Plot environment.
  #if args.mode == 'all':
      #ax.add_artist(plt.Circle(VIRTUAL_POSITION, VIRTUAL_RADIUS, color='green'))
  ax.add_artist(plt.Circle(CYLINDER_POSITION, CYLINDER_RADIUS, color='gray'))
  #ax.add_artist(plt.Circle(CYLINDER_POSITION2, CYLINDER_RADIUS, color='gray'))
  plt.plot([-WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, -WALL_OFFSET], 'k')
  plt.plot([-WALL_OFFSET, WALL_OFFSET], [WALL_OFFSET, WALL_OFFSET], 'k')
  plt.plot([-WALL_OFFSET, -WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')
  plt.plot([WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')

  # Plot a simple trajectory from the start position.
  # Uses Euler integration.
  dt = 0.01
  x = START_POSITION
  positions = [x]
  for t in np.arange(0., 20., dt):
    v = get_velocity(x, args.mode)
    x = x + v * dt
    positions.append(x)
  positions = np.array(positions)
  plt.plot(positions[:, 0], positions[:, 1], lw=2, c='r')

  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
  plt.ylim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
  plt.show()
