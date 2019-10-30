"""
OpenAI gym environment
"""

import gym


# #  environments:
environment_type = 'CartPole-v0'
# environment_type = 'MountainCar-v0'
# environment_type = 'MsPacman-v0'
#environment_type = 'Hopper-v2'



env = gym.make(environment_type)  # select environment
for i_episode in range(20):
    observation = env.reset()  # initialization 
    for t in range(100):
        env.render()                             # visualization 
        print(observation)                       # current state
        action = env.action_space.sample()       # current action
        observation, reward, done, info = env.step(action)     # simulate one step of the system with given actions
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()             #  quit environment




"""
.step() method 
    implementation of an agent-environment loop or closed control loop (control eng.)
returns 4 values:
    observation o : or measurements y(control eng.)
    reward R : or cost l (control eng.)
    done : flag that indicates the episode has terminated
    info : diagnostic information useful for debugging
"""

"""
.reset() method 
process gets started by calling reset(), which returns an initial observation
initialization function
"""

"""
.render() method 
visualization of the system
"""

"""
Action and State Spaces
"""
print(env.action_space)
print(env.observation_space)

"""
Action and State Spaces Bounds
"""
print(env.observation_space.high)
print(env.observation_space.low)

"""
full list of environments: https://gym.openai.com/envs/#classic_control
"""
from gym import envs
print(envs.registry.all())












