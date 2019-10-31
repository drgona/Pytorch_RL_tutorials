"""
DQN tutorial with Pytorch and OpenAI gym
source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


"""
environment setup
"""
# select cart pole environment (emulato model) from Gym
env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Replay memory
"""

# x: state
# u: actuon
# r: reward
# k: time instant
# dynamics: x_k+1 = f(k_x, u_k)
# transition = {x_k+1, x_k, u_k, r_k}
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Replay Memory - It stores the transitions that the agent observes, allowing us to reuse this data later.
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
"""
DQN model
"""    
# Q-learning: 1-step ahead learning
# R: return is a discounted, cumulative reward, or inverse loss function       
# R = sum(gamma^k*r_k)        
# objective is to maximize R    
# gamma: discount factor  gamma \in [0,1] 
# Q function is a mapping: {x_k, u_k} -> R     
# Q-learning approximated Q function as neural net
# V function: V(x_k+1) = max_u_k Q(x_k+1,u_k)
# policy: u = g(x_k)        
# training update rule is based on Bellman equation: Q(x_k,u_k) = r_k + gamma*Q(x_k+1,g(x_k+1))
# temporal difference (TD) error: delta =   Q(x_k,u_k) -  r_k - max_u_k gamma*Q(x_k+1,u_k)   
# the training objective is to minimize TD error by learning Q        
        
# in this case the  model is a convolutional neural network 
#that takes in the difference between the current and previous screen patches    
class DQN(nn.Module):

    def __init__(self, states, outputs, layers):
        super(DQN, self).__init__()
        # set of layers
        self.bn = nn.BatchNorm1d(states) # normalization     
        n_in =  states # number of inputs
        layerlist = [] #storing the layers
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            n_in = i
        #layerlist.append(nn.Linear(layers[-1],outputs)) #final layer        
         # assign layers to atributes
        self.layers = nn.Sequential(*layerlist)
        self.head = nn.Linear(layers[-1], outputs) # fully connected layer       

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):       
        x = self.bn(x)
        x = self.layers(x)            # apply layers
        return self.head(x)  
        #  return self.head(x.view(x.size(0), -1))  
            

    
    
"""
hyperparameters 
"""   

BATCH_SIZE = 128
GAMMA = 0.999        # discount
# epsilon is the probability of a random action sampling, otherwise a trained model will be used to generate action
EPS_START = 0.9         
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


# Get number of actions and observations from gym action space
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
layers = [10, 10]

# instantiates our model and its optimizer
policy_net = DQN(n_states, n_actions, layers).to(device)   # trained network
policy_net = policy_net.float()
target_net = DQN(n_states, n_actions, layers).to(device)   # target net is mostly fixed, updated by policy net every so often
target_net = target_net.float()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


for parameter in policy_net.parameters():
    print(parameter.size())

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

# select_action - will select an action accordingly to an epsilon greedy policy.
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        policy_net.eval()
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []

# a helper for plotting the durations of episodes,
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


"""
training
"""     
# single step of optimization
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    policy_net.eval()
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    target_net.eval()
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    policy_net.train()
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# main training loop
num_episodes = 500
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    state = torch.tensor(state)
    state = state.float().unsqueeze(0)
    
    screen = env.render(mode='rgb_array')

    for t in count(): # COUNTS episodes t before termination
        # Select and perform an action
        action = select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        next_state = torch.tensor(next_state)
        next_state = next_state.float().unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        
        screen = env.render(mode='rgb_array')

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()






     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    