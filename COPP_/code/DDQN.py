import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from envStreet import envStreet
import time

 
BATCH_SIZE = 16
LR = 0.01                    
GAMMA = 0.9                  
TARGET_REPLACE_ITER = 50    
MEMORY_CAPACITY = 500

ENV_A_SHAPE = 0  

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 32)
        self.fc1.weight.data.normal_(0, 1)    
        self.fc2 = nn.Linear(32, 32)
        self.fc2.weight.data.normal_(0, 1)    
        self.out = nn.Linear(32, N_ACTIONS)
        self.out.weight.data.normal_(0, 1)    

        self.fc_trans = nn.Linear(N_STATES, N_ACTIONS)  

    def forward(self, x):
        h1 = self.fc1(x)
        h1 = F.relu(h1)
        h2 = self.fc2(h1)
        h2 = F.relu(h2)
        actions_value = self.out(h2)  
         
        return actions_value

class DDQN(object):
    def __init__(self,N_STATES, N_ACTIONS):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        self.learn_step_counter = 0                                      
        self.memory_counter = 0                                          
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))      
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
    
    def save_model_increase(self):
        torch.save(self.eval_net.state_dict(),f'../Output_model/'+f'model_increase_{time.time()}')
    def save_model_reduce(self):
        torch.save(self.eval_net.state_dict(),f'../Output_model/'+f'model_reduce_{time.time()}')
    
    def choose_action(self, x, epsilon):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
         
        
        if np.random.uniform() > epsilon:    
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:    
            '''
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
            '''
             
            np.random.seed()
             
            action = np.random.randint(0, self.N_ACTIONS)
             
            
        return action

    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, a, r, s_))
         
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def get_attention_record(self, shape):
        self.attention_record.forward(shape)

    def learn(self):
         
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

         
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE, replace=False)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])
         
        q_eval = self.eval_net.forward(b_s).gather(1, b_a)   
        q_next = self.target_net(b_s_).detach()      
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)    
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss = loss
        
    def load_model_increase(self):
        self.eval_net.load_state_dict(torch.load('Output_model/'+'model_increase'))
    
    def load_model_reduce(self):
        self.eval_net.load_state_dict(torch.load('Output_model/'+'model_reduce'))
    
