from math import pi
from os import path
from typing import Counter
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import select
from pandas.core.indexes.base import ensure_index_from_sequences
from Environment import Environment, TGEnvironment
from OGP import OGP
from En import En
from copy import deepcopy
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import create_directory, plot_learning_curve, scale_action
from keras.utils import to_categorical
import torch
import networkx as nx
from dgl import DGLGraph
import random

from DDQN import DDQN
from envStreet import envStreet
from Q_table import Qtable



def run(envt, ogp, alg):

    REWARD = []
    SEARCH_TIME = []
    FAKE_REWARD = []

    RECORD_OBSERVATION = []

    remove_edge = [0,0]

    EPISODES = 200
    episode_reward = 0
    epsilon = 0.9   
    EPS_DECAY = 0.99
    envst = envStreet(envt)
    ACTION_LIST = list(np.arange(envt.action_low, envt.action_high, 0.5))
    N_ACTIONS = len(ACTION_LIST)

    street_id = np.array(list(range(len(envt.use_edge)))).reshape(-1,1)
    
    dic_one_hot_street_id = {}
    for index, one_hot in enumerate(to_categorical(street_id)):
        dic_one_hot_street_id[index] = one_hot[0].tolist()
    
    N_STATES = 5

    numpy_g = np.zeros((len(envt.use_edge),len(envt.use_edge)))
    for r,c in envt.use_edge:
        numpy_g[r,c] = 1
    
    nx_g = nx.Graph(numpy_g)
    g = DGLGraph(nx_g)

    dqn = GADQN(g, N_STATES, N_ACTIONS)
    
    SEARCH_TIME_ALL = []
    REWARD_ALL = []
    REWARD_AVG = []
    GTT_ALL = []

    AVG_reward_every_time = []
    dic_time_to_index = {}
    time_index = 0

    base_reward = -1

    s_mean = np.zeros(4)
    mean_N = 0

    for episode in range(EPISODES):

        if episode > 0:
             
            dqn.learn()

         
        obs = []
        
         
        En.envt = envt

        envt.temp_obs_ = [] 
        envt.temp_obs = []
        envt.temp_action = []
        envt.temp_action_ = []
        envt.dic_time_to_car = {}  
        envt.dic_time_to_global_travel_time = {}  
        envt.dic_time_to_query_time = {}  
        envt.dic_time_to_sas_ = {}  

        agents, SUM_START_TIME, first_node = envt.get_initial_query(envt.NUM_AGENTS)
        other_agents, _, _ = envt.get_initial_query_other(3*envt.NUM_AGENTS)

        H = deepcopy(agents)
        Q = deepcopy(other_agents)

        now_q = deepcopy(Q[-1])
        envt.current_time = now_q.now_t
        envt.last_time = envt.current_time

         
        envt.searchTime = {}  
        envt.searchTime_old = {}  
        envt.Street = {}  
        envt.history_all_car_num = []  
        envt.appear_car_num = 0  
        envt.L =  [[[] for i in range(envt.node_length)] for i in range(envt.node_length)]
        envt.delta_time = 0
        envt.query_gap = 0
        envt.query_count = 0
        envt.query_action = 1  
        final_Q = []
        all_final_Q = []  
        QUERY_TIME = 0  
        REWARD = []  
        REWARD_FAKE = []  
        QUERY_REWARD = []  
        QUERY_LIST = []  
        envt.initial_get_agent_number_of_edge()  
        observation_all_edge = np.zeros((len(envt.use_edge), N_STATES))

        POSITION_ACTION = []  

        print('Start plan ... ')
        
        pre_global_travel_time = 0

        LOAD_USE_EDGE = 0

        action_count = 0
        ALL_SEARCH_GAP = 0
        query_time_count = 0

        all_car_number_gap = 0
        car_number_count = 0

        now_appear_time = H[-1].belong_start_time
        dic_time_to_index[now_appear_time] = time_index
        
        while len(H) != 0:
            
            all_start_time = time.time()
            label = 0
            
            now_q_plan = H[-1]

            if len(Q) > 0:
                now_q_other = Q[-1]
                if now_q_other.now_t < now_q_plan.now_t:
                    label = 1
                    now_q = now_q_other
                else:
                    now_q = now_q_plan
            else:
                now_q = now_q_plan
            
            envt.current_time = now_q.now_t
             
            ILP_agents = [now_q]
            if label == 0:
                H.pop()
            else:
                Q.pop()
            
            alg_time = time.time()
            if label == 0:
                
                if alg == 1:
                    scored_final_actions, pre_one_travel_time = ogp.SBP(ILP_agents, remove_edge)  
                    if ILP_agents[0].belong_start_time == now_appear_time:
                        envt.dic_time_to_query_time[now_appear_time] += envt.delta_time
                        QUERY_TIME += envt.delta_time
                    else:
                        
                        envt.delta_time = 0
                        now_appear_time = ILP_agents[0].belong_start_time

                    pre_global_travel_time += pre_one_travel_time
                elif alg == 2:
                     
                     
                    scored_final_actions = ogp.Greedy(ILP_agents)
            if label == 1:
                 
                 
                scored_final_actions = ogp.SBP_other(ILP_agents)    

             

            other_time = time.time()
            
            for agent_id, action in enumerate(scored_final_actions):
                envt.update_agent_number_of_edge_add(ILP_agents[agent_id].now_node, action)

                 
                if (ILP_agents[agent_id].now_node, action) not in envt.all_car_number_of_edge:
                    envt.all_car_number_of_edge[(ILP_agents[agent_id].now_node, action)] = 1
                else:
                    envt.all_car_number_of_edge[(ILP_agents[agent_id].now_node, action)] += 1

                if (ILP_agents[agent_id].now_node, action) in envt.Street:
                    envt.Street[(ILP_agents[agent_id].now_node, action)].append(ILP_agents[agent_id])
                else:
                    envt.Street[(ILP_agents[agent_id].now_node, action)] = [ILP_agents[agent_id]]

                if len(ILP_agents[agent_id].pi) > 1:
                    envt.update_agent_number_of_edge_reduce(ILP_agents[agent_id].pi[-2], ILP_agents[agent_id].now_node)
                    envt.Street[(ILP_agents[agent_id].pi[-2], ILP_agents[agent_id].now_node)].remove(ILP_agents[agent_id])

            for update_agent_id, update_agent in enumerate(ILP_agents):
                
                agent_new = envt.update_agent(update_agent, scored_final_actions[update_agent_id])

                if label == 0:
                    if agent_new.now_node == agent_new.end_node:
                         
                         
                        final_Q.append(agent_new)
                        all_final_Q.append(agent_new)
                        envt.dic_time_to_car[agent_new.belong_start_time].remove(agent_new.agent_id)
                        if len(envt.dic_time_to_car[agent_new.belong_start_time]) == 0:
                            
                            global_travel_time = envt.dic_time_to_global_travel_time[agent_new.belong_start_time]
                            query_time = envt.dic_time_to_query_time[agent_new.belong_start_time]

                            if agent_new.belong_start_time not in dic_time_to_index:
                                time_index += 1
                                dic_time_to_index[agent_new.belong_start_time] = time_index

                            fake_reward = 0.000*envt.dic_time_to_query_time[agent_new.belong_start_time] + 1*envt.dic_time_to_global_travel_time[agent_new.belong_start_time]

                            if len(AVG_reward_every_time) <= dic_time_to_index[agent_new.belong_start_time]:
                                AVG_reward_every_time.append(fake_reward)
                            else:
                                AVG_reward_every_time[dic_time_to_index[agent_new.belong_start_time]] += fake_reward
                                AVG_reward_every_time[dic_time_to_index[agent_new.belong_start_time]] /= 2

                            reward = fake_reward - AVG_reward_every_time[dic_time_to_index[agent_new.belong_start_time]]
                             
                            reward = -reward
                            
                            if agent_new.belong_start_time in envt.dic_time_to_sas_:
                                for s,a,s_ in envt.dic_time_to_sas_[agent_new.belong_start_time]:
                                     
                                    dqn.store_transition(s, a, reward, s_)

                            del(envt.dic_time_to_car[agent_new.belong_start_time])
                            del(envt.dic_time_to_global_travel_time[agent_new.belong_start_time])
                            del(envt.dic_time_to_query_time[agent_new.belong_start_time])

                        else:
                            envt.dic_time_to_global_travel_time[agent_new.belong_start_time] += (agent_new.now_t - agent_new.appear_time)
                            
                    else:
                        envt.Binary_insert(H, agent_new)
                
                elif label == 1:
                    if agent_new.now_node == agent_new.end_node:
                        pass
                    else:
                        envt.Binary_insert(Q, agent_new)
            if len(H) == 0:
                break

             
            if label == 0 and envt.current_time - envt.last_time > envt.action_unit_time:  
                
                now_all_car_num = envt.get_all_car_num()
                envt.history_all_car_num.append(now_all_car_num)
                 
                 
                envt.all_car_number_of_edge.update((x,int(y*0.5)) for x, y in envt.all_car_number_of_edge.items())
                
                envt.temp_obs_ = []
                envt.temp_action_ = []

                for edge_count, edge in enumerate(envt.use_edge):
                    if envt.pre_car_number_of_edge[(edge[0], edge[1])] == 0:
                        query_beishu = 1
                        envt.pre_car_number_of_edge[(edge[0], edge[1])] = envt.car_number_of_edge[(edge[0], edge[1])]
                    else:
                        query_beishu = envt.action_unit_time * envt.car_number_of_edge[(edge[0], edge[1])]/envt.pre_car_number_of_edge[(edge[0], edge[1])]/(envt.current_time - envt.last_time)
                         
                        envt.pre_car_number_of_edge[(edge[0], edge[1])] = envt.car_number_of_edge[(edge[0], edge[1])] 

                    car_number_beishu = envt.car_number_of_edge[(edge[0], edge[1])]/now_all_car_num

                    observation = envt.node_attr_index[edge[0]]+[car_number_beishu, envt.car_number_of_edge[(edge[0], edge[1])], envt.Street_attr[(edge[0], edge[1])].Query_time_gap]  
                     
                    for o_index in range(len(observation)):
                        observation[o_index] = (observation[o_index]-observation_mean[o_index])/observation_std[o_index]

                     
                    if episode < 1:
                        RECORD_OBSERVATION.append(observation)

                     
                    observation_all_edge[edge_count] = observation
                 
                action_all = dqn.choose_action(observation_all_edge, epsilon)

                 
                envt.temp_obs_ = observation_all_edge.copy()

                for edge_count, edge in enumerate(envt.use_edge):
                     

                    final_action = ACTION_LIST[action_all[edge_count][0]]
                    envt.temp_action_.append([action_all[edge_count][0]]) 

                    if query_beishu > final_action:
                        envt.Street_attr[(edge[0], edge[1])].Query_state = 1
                        envt.Street_attr[(edge[0], edge[1])].Query_time_gap = 0
                    else:
                        envt.Street_attr[(edge[0], edge[1])].Query_state = 0
                        envt.Street_attr[(edge[0], edge[1])].Query_time_gap += (envt.current_time - envt.last_time)
                
                if len(envt.temp_obs) != 0:
                    if agent_new.belong_start_time not in envt.dic_time_to_sas_:
                        envt.dic_time_to_sas_[agent_new.belong_start_time] = [[envt.temp_obs, envt.temp_action, envt.temp_obs_]]
                    else:
                        envt.dic_time_to_sas_[agent_new.belong_start_time].append([envt.temp_obs, envt.temp_action, envt.temp_obs_])
                
                 
                envt.temp_obs = envt.temp_obs_.copy()
                envt.temp_action = envt.temp_action_.copy()

                envt.last_time = envt.current_time
             
        epsilon *= EPS_DECAY
         
        final_sum_travel_time = 0
        final_deal_travel_time = 0

        for q in all_final_Q:
            if q.belong_start_time != now_appear_time:
                final_sum_travel_time += (q.now_t - q.appear_time)
                final_deal_travel_time += envt.min_travel_time[q.start_node, q.end_node]

        REWARD_ALL.append(-(0.000*QUERY_TIME + 1*final_sum_travel_time))
        SEARCH_TIME_ALL.append(QUERY_TIME)
        GTT_ALL.append(final_sum_travel_time)
        
    return final_Q,[]

if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser('Setting Parameters')
    parser.add_argument('-n','--NUM_AGENTS',type=int,default=5000,help='Number of Query')
    parser.add_argument('-s','--SEED',type=int,default=20,help='random seed')
    parser.add_argument('-alg','--alg',type=int,default=1,help='choose algorithm')

    args = parser.parse_args()
    envt = TGEnvironment(args.NUM_AGENTS, args.SEED)
    ogp = OGP(envt)
    run(envt, ogp, args.alg)