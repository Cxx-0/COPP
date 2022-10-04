import numpy as np 
import random

class Qtable(object):

    def __init__(self, N_STATES, N_ACTIONS):
        self.gap_size = 100
        self.obs_size = [300, 300]  
        self.Q_table_size = self.obs_size + [N_ACTIONS]
        self.Q_table = np.random.uniform(0,0,size = self.Q_table_size)
        self.DISCOUNT = 0.95  
        self.LEARNINT_RATE = 0.01  
        self.N_ACTIONS = N_ACTIONS

    def get_qtable_position(self, status):
        
        position = np.array([status[0]/self.gap_size, status[1]*10])
         

        return tuple(position.astype(np.int))

    def choose_action(self, position, epsilon):
    
        if np.random.uniform() > epsilon:    
            action = np.argmax(self.Q_table[position])
        else:  
            np.random.seed()
            action = np.random.randint(0, self.N_ACTIONS)

        return action

    def update_Qtable(self, position, action, q_future_max, q_current, reward, episode):

         
        self.LEARNINT_RATE = 1/(500+episode)  
        self.Q_table[position+(action,)] = (1-self.LEARNINT_RATE)*q_current + self.LEARNINT_RATE*(reward + self.DISCOUNT*q_future_max) 
         

    def update_Qtable_reply(self, PA, reward_all, episode):

        self.LEARNINT_RATE = 1/(500+episode)
        for position_action in PA:
            self.Q_table[position_action] = (1-self.LEARNINT_RATE)*self.Q_table[position_action] + self.LEARNINT_RATE*(reward_all) 
    
    def save_Qtable(self):

        np.save('Qtable.npy', self.Q_table)

    def load_Qtable(self):

        self.Q_table = np.load('Qtable.npy')