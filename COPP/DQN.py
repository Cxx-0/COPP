import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from envStreet import envStreet
import time

# Hyper Parameters
BATCH_SIZE = 16
LR = 0.01                   # learning rate
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 50   # target update frequency
MEMORY_CAPACITY = 500

ENV_A_SHAPE = 0 #if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 32)
        self.fc1.weight.data.normal_(0, 1)   # initialization
        self.fc2 = nn.Linear(32, 32)
        self.fc2.weight.data.normal_(0, 1)   # initialization
        self.out = nn.Linear(32, N_ACTIONS)
        self.out.weight.data.normal_(0, 1)   # initialization

        self.fc_trans = nn.Linear(N_STATES, N_ACTIONS) # 用于残差连接

    def forward(self, x):
        h1 = self.fc1(x)
        h1 = F.relu(h1)
        h2 = self.fc2(h1)
        h2 = F.relu(h2)
        actions_value = self.out(h2) #+ self.fc_trans(x)
        #actions_value = F.relu(h3)
        return actions_value

class DQN(object):
    def __init__(self,N_STATES, N_ACTIONS):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
    
    def save_model_increase(self):
        torch.save(self.eval_net.state_dict(),f'model_increase_{time.time()}')
    def save_model_reduce(self):
        torch.save(self.eval_net.state_dict(),f'model_reduce_{time.time()}')
    
    def choose_action(self, x, epsilon):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        
        if np.random.uniform() > epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:   # random
            '''
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
            '''
            #print(x,action)
            np.random.seed()
            #action_list_temp = [0, 1,1,1, 2,2,2,2,2,2]
            action = np.random.randint(0, self.N_ACTIONS)
            #action = np.random.choice(action_list_temp)
            
        return action

    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, a, r, s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def get_attention_record(self, shape):
        self.attention_record.forward(shape)

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE, replace=False)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net.forward(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss = loss
        
    def load_model_increase(self):
        self.eval_net.load_state_dict(torch.load('model_increase_1'))
    
    def load_model_reduce(self):
        self.eval_net.load_state_dict(torch.load('model_reduce_1'))
    
