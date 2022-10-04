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

from DDQN import DDQN
from envStreet import envStreet
from Q_table import Qtable



def run(envt, ogp, alg):

    REWARD = []
    SEARCH_TIME = []
    REWARD_EVERY_SUM =[]
    FAKE_REWARD = []
    RECORD_OBSERVATION = []
    ALL_CAR_NUM = []
    ALL_CAR_NUM_E0 = []
    QUERY_TIME_E0 = []

    remove_edge = [0,0]

    EPISODES = 1000
    episode_reward = 0
    epsilon = 0.9   
    EPS_DECAY = (0.05/epsilon)**(1/EPISODES)
    envst = envStreet(envt)
    ACTION_LIST_increase = [1.5, 2.5, 3.5]
    ACTION_LIST_reduce = [0.3, 0.6, 0.9]
    N_ACTIONS = len(ACTION_LIST_increase)

    street_id = np.array(list(range(len(envt.use_edge)))).reshape(-1,1)
    
    dic_one_hot_street_id = {}
    for index, one_hot in enumerate(to_categorical(street_id)):
        dic_one_hot_street_id[index] = one_hot[0].tolist()
    
    N_STATES = 3

    dqn_increase = DDQN(N_STATES, N_ACTIONS)
    dqn_reduce = DDQN(N_STATES, N_ACTIONS)

    SEARCH_TIME_ALL = []
    REWARD_ALL = []
    REWARD_AVG = []
    GTT_ALL = []
    
    AVG_reward_every_time = []
    dic_time_to_index = {}

    base_reward = -1

    s_mean = np.zeros(4)
    mean_N = 0

    for episode in range(EPISODES):

        REWARD_EVERY = []
        GTT_ALL_part = []
        ALL_CAR_NUM_part = []
        REWARD_ALL_part = []
        value_gap = []
        value_gap_count = 0

        if episode > 1:
            dqn_increase.learn()
            dqn_reduce.learn

        obs = []
        
        En.envt = envt

        envt.initial_para() 

        agents, SUM_START_TIME, first_node = envt.get_initial_query(envt.NUM_AGENTS)
        other_agents, _, _ = envt.get_initial_query_other(3*envt.NUM_AGENTS)

        H = deepcopy(agents)
        Q = deepcopy(other_agents)

        now_q = deepcopy(Q[-1])
        envt.current_time = now_q.now_t
        envt.last_time = envt.current_time

        QUERY_TIME = 0 

        envt.L =  [[[] for i in range(envt.node_length)] for i in range(envt.node_length)]
        final_Q = []
        all_final_Q = [] 
        REWARD = [] 
        REWARD_FAKE = [] 
        QUERY_REWARD = [] 
        QUERY_LIST = [] 
        envt.initial_get_agent_number_of_edge() 
        observation_all_edge = np.zeros((len(envt.use_edge), N_STATES))

        POSITION_ACTION = [] 

        for edge in envt.use_edge: 
            envt.Street_attr[(edge[0], edge[1])].Query_state = 0

        print('Start plan ... ')
        
        pre_global_travel_time = 0

        LOAD_USE_EDGE = 0

        action_count = 0
        ALL_SEARCH_GAP = 0
        query_time_count = 0

        all_car_number_gap = 0
        car_number_count = 0

        now_appear_time = H[-1].belong_start_time
        time_index = 0
        dic_time_to_index[now_appear_time] = time_index

        r__ = 1.05
        reduce_car_num = 0
        GLOBAL_TRAVEL_TIME = 0
        
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
                    else:
                        now_appear_time = ILP_agents[0].belong_start_time

                    pre_global_travel_time += pre_one_travel_time

            if label == 1:
                 
                 
                scored_final_actions = ogp.SBP_other(ILP_agents)    


            other_time = time.time()
            
            for agent_id, action in enumerate(scored_final_actions):
                envt.update_agent_number_of_edge_add(ILP_agents[agent_id].now_node, action)
                
                if label == 0:
                    envt.car_number_of_edge_my_plot[ILP_agents[agent_id].now_node, action] += 1

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
                    if label == 0:
                        envt.car_number_of_edge_my_plot[ILP_agents[agent_id].pi[-2], ILP_agents[agent_id].now_node] -= 1

            for update_agent_id, update_agent in enumerate(ILP_agents):
                
                agent_new, delta_t = envt.update_agent(update_agent, scored_final_actions[update_agent_id])
                
                if label == 0:
                    reduce_car_num += delta_t
                    if agent_new.now_node == agent_new.end_node:

                        envt.update_agent_number_of_edge_reduce(agent_new.pi[-2], agent_new.pi[-1])
                        envt.car_number_of_edge_my_plot[agent_new.pi[-2], agent_new.pi[-1]] -= 1

                        final_Q.append(agent_new)
                        all_final_Q.append(agent_new)
                        envt.dic_time_to_car[agent_new.belong_start_time].remove(agent_new.agent_id)
                        
                        if len(envt.dic_time_to_car[agent_new.belong_start_time]) == 0:
                            
                            global_travel_time = envt.dic_time_to_global_travel_time[agent_new.belong_start_time]
                            query_time = envt.dic_time_to_query_time[agent_new.belong_start_time]
                            if agent_new.belong_start_time not in dic_time_to_index:
                                time_index += 1
                                dic_time_to_index[agent_new.belong_start_time] = time_index

                            fake_reward = 0*query_time + 1*global_travel_time

                            if episode == 0:
                                reward = 0
                                AVG_reward_every_time.append(fake_reward)
                                GTT_ALL_part.append(fake_reward)
                            else:
                                reward = fake_reward - AVG_reward_every_time[dic_time_to_index[agent_new.belong_start_time]]

                                reward = -1*reward
                                REWARD_EVERY.append(reward)
                                GTT_ALL_part.append(fake_reward)

                            del(envt.dic_time_to_car[agent_new.belong_start_time])
                            del(envt.dic_time_to_global_travel_time[agent_new.belong_start_time])
                            del(envt.dic_time_to_query_time[agent_new.belong_start_time])

                        else:
                            envt.dic_time_to_global_travel_time[agent_new.belong_start_time] += (agent_new.now_t - agent_new.appear_time)
                        
                    else:
                        envt.Binary_insert(H, agent_new)
                
                elif label == 1:
                    if agent_new.now_node == agent_new.end_node:
                        envt.update_agent_number_of_edge_reduce(agent_new.pi[-2], agent_new.pi[-1])
                        pass
                    else:
                        envt.Binary_insert(Q, agent_new)

            if label == 0 and envt.current_time - envt.last_time > envt.action_unit_time: 
                
                now_all_car_num = envt.get_all_car_num()
                envt.history_all_car_num.append(now_all_car_num)
                envt.all_car_number_of_edge.update((x,int(y*0.5)) for x, y in envt.all_car_number_of_edge.items())
                
                envt.temp_obs_ = []
                envt.temp_action_ = []
                edge_count = 0

                QUERY_TIME += envt.delta_time 
                now_all_car_num_ = reduce_car_num 
                GLOBAL_TRAVEL_TIME += reduce_car_num
                reduce_car_num = 0
                r__*=1.05
                ALL_CAR_NUM_part.append(now_all_car_num_)
                CAR_NUM_index = ALL_CAR_NUM_part.index(now_all_car_num_)

                if len(ALL_CAR_NUM_E0) <= CAR_NUM_index:
                    FIRST_CAR_NUM = now_all_car_num_
                    FIRST_CAR_NUM_std = 1
                    FIRST_CAR_NUM_mean = 1
                else:
                    FIRST_CAR_NUM = ALL_CAR_NUM_E0[CAR_NUM_index]
                    FIRST_CAR_NUM_std = np.std(ALL_CAR_NUM_E0)+1
                    FIRST_CAR_NUM_mean = np.mean(ALL_CAR_NUM_E0)+1
                
                if len(QUERY_TIME_E0) <= CAR_NUM_index:
                    FIRST_QUERY_TIME = envt.delta_time
                    FIRST_QUERY_TIME_std = 1
                    FIRST_QUERY_TIME_mean = 1
                else:
                    FIRST_QUERY_TIME = QUERY_TIME_E0[CAR_NUM_index]
                    FIRST_QUERY_TIME_std = np.std(QUERY_TIME_E0)+1
                    FIRST_QUERY_TIME_mean = np.mean(QUERY_TIME_E0)+1

                if episode == 1:
                    ALL_CAR_NUM_E0.append(now_all_car_num_)
                    QUERY_TIME_E0.append(envt.delta_time)

                FIRST_REWARD = 0.999*FIRST_CAR_NUM+0.001*FIRST_QUERY_TIME

                real_reward = 0.999*now_all_car_num_ + 0.001*envt.delta_time
                
                if episode == 0:
                    real_reward = 0
                else:
                    real_reward -= FIRST_REWARD
                
                real_reward = -real_reward
                REWARD_ALL_part.append(real_reward)

                for edge_count, edge in enumerate(envt.use_edge):

                    car_number_beishu = envt.car_number_of_edge[(edge[0], edge[1])]/now_all_car_num

                    observation = envt.node_attr_index[edge[0]]+[envt.car_number_of_edge[(edge[0], edge[1])]]
                    if episode < 1:
                        RECORD_OBSERVATION.append(observation)

                    observation_all_edge[(edge_count)] = observation

                for edge_count, edge in enumerate(envt.use_edge):

                    if (edge[0], edge[1]) in envt.car_num_avg and envt.car_num_avg[(edge[0], edge[1])] != []:
                        query_beishu = envt.action_unit_time * (np.mean(envt.car_num_avg[(edge[0], edge[1])])+1)/(envt.pre_car_number_of_edge[(edge[0], edge[1])]+1)/(envt.current_time - envt.last_time)
                        envt.pre_car_number_of_edge[(edge[0], edge[1])] = np.mean(envt.car_num_avg[(edge[0], edge[1])])
                        envt.car_num_avg[(edge[0], edge[1])] = []
                    else:
                        query_beishu = 1
                    
                    observation_ = observation_all_edge[edge_count]
                    
                    envt.temp_obs_.append(observation_)

                    if query_beishu < 1:
                        final_action = dqn_reduce.choose_action(observation_, epsilon)
                        envt.temp_action_.append(final_action)
                        final_action = ACTION_LIST_reduce[final_action]
                    else:
                        final_action = dqn_increase.choose_action(observation_, epsilon)
                        envt.temp_action_.append(final_action)
                        final_action = ACTION_LIST_increase[final_action]
                    
                    if query_beishu < 1:
                        if final_action < 1 and query_beishu < final_action:
                            envt.Street_attr[(edge[0], edge[1])].Query_state = 1
                            envt.Street_attr[(edge[0], edge[1])].Query_time_gap = 0
                        else:
                            envt.Street_attr[(edge[0], edge[1])].Query_state = 0
                            envt.Street_attr[(edge[0], edge[1])].Query_time_gap += (envt.current_time - envt.last_time)
                    else:
                        if final_action > 1 and query_beishu > final_action:
                            envt.Street_attr[(edge[0], edge[1])].Query_state = 1
                            envt.Street_attr[(edge[0], edge[1])].Query_time_gap = 0
                        else:
                            envt.Street_attr[(edge[0], edge[1])].Query_state = 0
                            envt.Street_attr[(edge[0], edge[1])].Query_time_gap += (envt.current_time - envt.last_time)
                    
                    if len(envt.temp_obs) != 0 and episode > 1:
                        if agent_new.belong_start_time not in envt.dic_time_to_sas_:
                            envt.dic_time_to_sas_[agent_new.belong_start_time] = [[envt.temp_obs[edge_count], envt.temp_action[edge_count], envt.temp_obs_[edge_count]]]
                        else:
                            envt.dic_time_to_sas_[agent_new.belong_start_time].append([envt.temp_obs[edge_count], envt.temp_action[edge_count], envt.temp_obs_[edge_count]])

                    if len(envt.temp_obs) != 0 and episode > 1:
                        if query_beishu < 1:
                            dqn_reduce.store_transition(envt.temp_obs[edge_count], envt.temp_action[edge_count], real_reward, envt.temp_obs_[edge_count])
                        else:
                            dqn_increase.store_transition(envt.temp_obs[edge_count], envt.temp_action[edge_count], real_reward, envt.temp_obs_[edge_count])

                    edge_count += 1

                envt.temp_obs = deepcopy(envt.temp_obs_)
                envt.temp_action = deepcopy(envt.temp_action_)

                envt.last_time = envt.current_time
                envt.delta_time = 0

            if len(H) == 0:
                break
        
         
        if episode % 50 == 0:
            dqn_increase.save_model_increase()
            dqn_reduce.save_model_reduce()
            
        epsilon *= EPS_DECAY
        final_sum_travel_time = 0
        final_deal_travel_time = 0
        
        final_sum_travel_time = GLOBAL_TRAVEL_TIME
        for q in all_final_Q:                
            final_deal_travel_time += envt.min_travel_time[q.start_node, q.end_node]
        
        print('GTT:', final_sum_travel_time)
        print('Ideal_GTT', final_deal_travel_time)
        print('TEC', QUERY_TIME)

        if episode > 0:
            REWARD_ALL.append(np.sum(REWARD_ALL_part))
            SEARCH_TIME_ALL.append(QUERY_TIME)
            GTT_ALL.append(final_sum_travel_time)
            REWARD_EVERY_SUM.append(np.sum(REWARD_EVERY))
            ALL_CAR_NUM.append(np.sum(ALL_CAR_NUM_part))
    
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