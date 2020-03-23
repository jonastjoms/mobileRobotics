import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

# Load rewards from training and testing
training_rewards = pickle.load( open("/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/training_rewards.p", "rb" ) )
error = pickle.load( open("/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/errors.p", "rb" ) )

#print(training_rewards)
#print(errors)
rewards =  []
errors = []
for key in training_rewards:
    if type(key) == int:
        if key == 60:
            continue
        rewards.append(training_rewards[key])
        errors.append(error[key])

plt.figure()
plt.plot(np.arange(len(rewards)), rewards, linewidth = 6)
plt.title("Learning curve related to average reward per episode", fontsize = 14)
plt.xlabel("Number of episodes", fontsize = 14)
plt.ylabel("Reward", fontsize = 14)

plt.figure()
plt.plot(np.arange(len(errors)), errors, linewidth = 6)
plt.title("Learning curve related to average error per episode", fontsize = 14)
plt.xlabel("Number of episodes", fontsize = 14)
plt.ylabel("Error", fontsize = 14)

training_rewards = pickle.load( open("/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/training_rewards_extension.p", "rb" ) )
error = pickle.load( open("/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/errors_extension.p", "rb" ) )
p_values = pickle.load( open("/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/p_values.p", "rb" ) )
d_values = pickle.load( open("/home/jonas/catkin_ws/src/exercises/part2/ros/checkpoints/d_values.p", "rb" ) )

#print(training_rewards)
#print(errors)
#print(p_values)

rewards =  []
errors = []
p = []
d = []
for key in training_rewards:
    if type(key) == int:
        rewards.append(training_rewards[key])
        errors.append(error[key])
        p.append(p_values[key])
        d.append(d_values[key])
#
# plt.figure()
# plt.plot(np.arange(len(rewards)), rewards)
#
# plt.figure()
# plt.plot(np.arange(len(errors)), errors)

plt.figure()
plt.plot(np.arange(len(rewards)), p, linewidth = 6)
plt.title("Change in Kp parameter of PD-controller", fontsize = 14)
plt.xlabel("Number of episodes", fontsize = 14)
plt.ylabel("Kp parameter", fontsize = 14)

plt.figure()
plt.plot(np.arange(len(errors)), d, linewidth = 6)
plt.show()
plt.title("Change in Kd parameter of PD-controller", fontsize = 14)
plt.xlabel("Number of episodes", fontsize = 14)
plt.ylabel("Kd parameter", fontsize = 14)
