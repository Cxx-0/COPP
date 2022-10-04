from Query import Query
from Street import Street_attr
from abc import ABCMeta, abstractmethod
import numpy as np
from typing import List
import random
import math
import pandas as pd
import scipy.sparse as ss
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None

inf = 99999999

class Environment(metaclass=ABCMeta):

    def __init__(self, DATA_DIR: str, car_speed: int, alpha: float, NUM_AGENTS: int, delta_time: float, SEED: int) -> None:
         
        self.DATA_DIR = DATA_DIR
        self.car_speed = car_speed
        self.alpha = alpha 
        self.NUM_AGENTS = NUM_AGENTS
        self.delta_time = delta_time
        self.SEED = SEED

        self.current_time: float = 0.0
        self.last_time: float = 0.0
        self.same_time_gap = {}  
        self.initial_same_time_gap_constant = 0.001  
        self.distance_low_gap_percent: float = 10
        self.distance_high_gap_percent: float = 10
        self.distance_range_gap = 10  
        self.learning_rate_distance_range_gap = 2
        self.uprate_of_distance = 0.05
        self.unit_time: float = 0.08333  
        self.action_unit_time: float = 0.08333  
        self.r_: float = 0.95  
        self.global_query_beishu = 0  

        self.action_low = 0.5  
        self.action_high = 3.5  

        self.searchTime = {}  
        self.searchTime_old = {}  
        self.Street = {}  
        self.Street_attr = {}  
        self.history_all_car_num = []  
        self.node_attr_index = {}
        self.appear_car_num = 0  

        self.dic_time_to_car = {}  
        self.dic_time_to_global_travel_time = {}  
        self.dic_time_to_query_time = {}  
        self.dic_time_to_sas_ = {}  
        self.dic_car_number_pre = {}  

         
        self.temp_obs_ = [] 
        self.temp_obs = []
        self.temp_action = []
        self.temp_action_ = []

         
        self.query_gap = 0  
        self.query_count = 0  
        self.query_action = 0  
        self.query_action_old = 0  

    @abstractmethod
    def initialise_environment(self):
        raise NotImplementedError

class TGEnvironment(Environment):
                                                        
    def __init__(self, NUM_AGENTS, SEED, DATA_DIR: str='../sample_data/', car_speed: int = 674, alpha: float = 0.1, delta_time: float = 0.0):   
        super().__init__(DATA_DIR=DATA_DIR, car_speed=car_speed, alpha=alpha, NUM_AGENTS=NUM_AGENTS, delta_time=delta_time,SEED=SEED)

        self.initialise_environment() 

    def initialise_environment(self):
        print('Loading Environment...')
        
        SHORTESTDISTANCE_FILE: str = self.DATA_DIR + 'shortest_min_mat.npy'
         

        self.shortest_distance = np.load(SHORTESTDISTANCE_FILE)

        graph = pd.read_csv(self.DATA_DIR + 'edge.csv')
        self.node_attr = pd.read_csv(self.DATA_DIR + 'node.csv')
        node_length = len(pd.Series(list(graph['start_node'])+list(graph['end_node'])).drop_duplicates())+3
        row_all = list(graph.start_node)+list(graph.end_node)
        col_all = list(graph.end_node)+list(graph.start_node)
        data_ = list(graph.distance)+list(graph.distance)
        self.TG_graph_adj = ss.coo_matrix((data_, (row_all, col_all)),shape=(node_length, node_length)).tocsr()
        
        for i in range(self.TG_graph_adj.shape[0]):
            self.TG_graph_adj[i, i] = inf
        self.Predecessors = np.load(self.DATA_DIR+'Predecessors.npy')
        self.dontUseNode = []
        
         
        self.min_travel_time = self.shortest_distance/self.car_speed

        self.street_capacity = self.min_travel_time*45000/10  

        self.mean_distance_of_edge = self.get_mean_distance_of_edge()

        self.mean_travel_time = self.mean_distance_of_edge/self.car_speed

        self.use_node = list(range(self.TG_graph_adj.shape[0]))

        data = []
        row = []
        col = []
        self.node_length = self.TG_graph_adj.shape[0]
        self.car_number_of_edge = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()
        self.car_number_of_edge_my_plot = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()  
        self.pre_car_number_of_edge = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()
        self.ganzhi_car_number_of_edge = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()
        self.all_car_number_of_edge = {}  
        self.car_num_avg = {}  
        
        self.other_car_number_of_edge = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()

        self.index_array = np.array(range(self.TG_graph_adj.shape[0])) 

        self.adj_node_dic = self.initial_adj_node_dic()
        np.save(self.DATA_DIR+'adj_node_dic.npy', self.adj_node_dic)
         

        self.L =  [[[] for i in range(self.node_length)] for i in range(self.node_length)]
         

         
        self.use_edge = np.load(self.DATA_DIR+'use_edge.npy', allow_pickle=True)[:50]

        self.use_edge_list = self.use_edge.tolist()  
        self.use_node_H = []

        self.lowest_distance = 100
        self.highest_distance = 2000

        self.initial_Street_attr()  
        self.initial_node_attr_index()  
        self.initial_centrality()  
        self.initial_same_time_gap()  

    def initial_node_attr_index(self):

        for index, value in self.node_attr.iterrows():
            self.node_attr_index[value.ID] = [value.X_scale, value.Y_scale]
    
    def initial_centrality(self):
        pass

    def initial_para(self):
        self.current_time: float = 0.0
        self.last_time: float = 0.0
        self.distance_low_gap_percent: float = 10
        self.distance_high_gap_percent: float = 10
        self.distance_range_gap = 10  
        self.learning_rate_distance_range_gap = 0.05
        self.uprate_of_distance = 0.05
        self.unit_time: float = 0.08333  
        self.action_unit_time: float = 0.08333  
        self.r_: float = 0.95  
        self.global_query_beishu = 0  

        self.action_low = 0.5  
        self.action_high = 3.5  

        self.searchTime = {}  
        self.searchTime_old = {}  
        self.Street = {}  
        self.Street_attr = {}  
        self.history_all_car_num = []  
        self.node_attr_index = {}
        self.appear_car_num = 0  

        self.dic_time_to_car = {}  
        self.dic_time_to_global_travel_time = {}  
        self.dic_time_to_query_time = {}  
        self.dic_time_to_sas_ = {}  
        self.dic_car_number_pre = {}  

         
        self.temp_obs_ = [] 
        self.temp_obs = []
        self.temp_action = []
        self.temp_action_ = []

         
        self.query_gap = 0  
        self.query_count = 0  
        self.query_action = 0  
        self.query_action_old = 0  

        self.initial_Street_attr()  
        self.initial_node_attr_index()  
        self.initial_centrality()  

    def initial_same_time_gap(self):
        for node1, node2 in self.use_edge:
            if node1 not in self.same_time_gap:
                self.same_time_gap[node1] = self.initial_same_time_gap_constant
            if node2 not in self.same_time_gap:
                self.same_time_gap[node2] = self.initial_same_time_gap_constant

    def get_initial_query(self, NUM_AGENTS: int) -> List[Query]:
        
        first_node_ = 0
        random.seed(self.SEED)
        initial_query = []
        first_node_ = random.sample(list(self.use_node), 1)[0]

         
        
        first_node = [first_node_]
        self.use_node_1 = [first_node_]

        self.use_node_1 = list(range(0,200))
        dis = 0
        for i in range(NUM_AGENTS):
            a = random.sample(self.use_node_1, 1)[0]
            b = random.sample(self.use_node_1, 1)[0]   
            
            while (a == b or self.shortest_distance[a][b] < self.lowest_distance or self.shortest_distance[a][b] > self.highest_distance):  
                b = random.sample(self.use_node_1, 1)[0]
                if self.shortest_distance[a,b] == np.inf:
                    a = random.sample(self.use_node_1, 1)[0]

            dis += self.shortest_distance[a,b]
            initial_query.append([a, b])
        
        np.random.seed(self.SEED)
        query_time = np.array(list(np.random.normal(6.3,0.1,int(NUM_AGENTS/2))) + list(np.random.normal(6.8,0.1,int(NUM_AGENTS/2))))
         
        Q = [Query(initial_query[i][0], initial_query[i][1],
         initial_query[i][0], query_time[i], [initial_query[i][0]], i, [initial_query[i][0]], query_time[i], 0) for i in range(NUM_AGENTS)]

        Q = sorted(Q, reverse=True, key=lambda keys:keys.now_t)

        temp_time = Q[-1].appear_time
        self.dic_time_to_car[temp_time] = [Q[-1].agent_id]
        self.dic_time_to_global_travel_time[temp_time] = 0
        self.dic_time_to_query_time[temp_time] = 0
        Q[-1].belong_start_time = temp_time
        for agent in Q[-2::-1]:
            if agent.appear_time - temp_time <= self.unit_time:
                self.dic_time_to_car[temp_time].append(agent.agent_id)
                agent.belong_start_time = temp_time
            else:
                temp_time = agent.appear_time
                agent.belong_start_time = temp_time
                self.dic_time_to_car[temp_time] = [agent.agent_id]
                self.dic_time_to_global_travel_time[temp_time] = 0
                self.dic_time_to_query_time[temp_time] = 0

        SUM_START_TIME = 0
        for q in Q:
            SUM_START_TIME += q.now_t

        return Q, SUM_START_TIME,first_node_

    def get_initial_query_other(self, NUM_AGENTS: int) -> List[Query]:
        
        first_node_ = 0
        random.seed(self.SEED+20)
        initial_query = []

        for i in range(NUM_AGENTS):
            a = random.sample(self.use_node_1, 1)[0]
            b = random.sample(self.use_node_1, 1)[0]   

            while (a == b or self.shortest_distance[a][b] < self.lowest_distance or self.shortest_distance[a][b] > self.highest_distance):  
                b = random.sample(self.use_node_1, 1)[0]
                if self.shortest_distance[a,b] == np.inf:
                    a = random.sample(self.use_node_1, 1)[0]
                
            initial_query.append([a, b])   
            
        np.random.seed(self.SEED+20) 
        query_time = np.array(list(np.random.normal(6.3,0.1,int(NUM_AGENTS/2))) + list(np.random.normal(6.8,0.1,int(NUM_AGENTS/2))))
         
        Q = [Query(initial_query[i][0], initial_query[i][1],
         initial_query[i][0], query_time[i], [initial_query[i][0]], i, [initial_query[i][0]], query_time[i], 1) for i in range(NUM_AGENTS)]

        Q = sorted(Q, reverse=True, key=lambda keys:keys.now_t)

        SUM_START_TIME = 0
        for q in Q:
            SUM_START_TIME += q.now_t

        return Q, SUM_START_TIME,first_node_

    def initial_adj_node_dic(self):
        adj_node_dic = {}

        for node in self.use_node:
            adj_node_dic[node] = self.TG_graph_adj[node].indices

        return adj_node_dic

    def get_adj_node(self, source: int) -> List[int]:

        return self.adj_node_dic[source]

    def get_mean_distance_of_edge(self) -> float:
        mean_distance_of_edge = 15.874  
        return mean_distance_of_edge

    def get_shortest_distance(self, source: int, destination: int) -> int:
        return self.shortest_distance[source, destination]

    def get_min_travel_time(self, source: int, destination: int) -> int:
        return self.min_travel_time[source, destination]

    def get_edge_travel_time_by_edge_density_nonlinear(self, min_travel_time, flow) -> float:
        
        b = min_travel_time/self.mean_travel_time
        
        if b < 0.25:
            C = 5
        elif b < 0.5:
            C = 10
        elif b < 0.75:
            C = 15  
        elif b < 1:
            C = 20  
        elif b < 1.25:
            C = 25  
        elif b < 1.5:
            C = 30  
        elif b < 1.75:
            C = 35  
        elif b < 2:
            C = 40  
        else:
            C = 50  
        
         
        travel_time = min_travel_time * (1+1*(flow/C)**2)

        return travel_time

    def get_agent_number_of_edge(self, source: int, destination: int) -> int:
        return self.car_number_of_edge[source, destination]

    def update_agent_number_of_edge_add(self, source: int, destination: int) -> None:
        self.car_number_of_edge[source, destination] += 1
        if (source, destination) in self.car_num_avg:
            self.car_num_avg[(source, destination)].append(self.car_number_of_edge[source, destination])
        else:
            self.car_num_avg[(source, destination)] = [self.car_number_of_edge[(source, destination)]]

    def update_agent_number_of_edge_reduce(self, source: int, destination: int) -> None:
        self.car_number_of_edge[source, destination] -= 1
        if self.car_number_of_edge[source, destination] < 0:
            print(self.car_number_of_edge[source, destination])
        if (source, destination) in self.car_num_avg:
            self.car_num_avg[(source, destination)].append(self.car_number_of_edge[source, destination])
        else:
            self.car_num_avg[(source, destination)] = [self.car_number_of_edge[(source, destination)]]

    def initial_get_agent_number_of_edge(self):

        data = []
        row = []
        col = []
        self.node_length = self.TG_graph_adj.shape[0]
        self.car_number_of_edge = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()
        self.car_number_of_edge_my_plot = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()  
        self.pre_car_number_of_edge = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()
        self.ganzhi_car_number_of_edge = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()
        self.all_car_number_of_edge = {}  
        self.car_num_avg = {}  

    def get_real_travel_time_of_node_to_node(self, agent, action) -> float:
        flow = self.car_number_of_edge[agent.now_node, action]
        min_travel_time = self.min_travel_time[agent.now_node][action]
        t_va_v = self.get_edge_travel_time_by_edge_density_nonlinear(min_travel_time, flow)

        return t_va_v

    def update_agent(self, agent, action) -> Query:
        
        t_va_v = self.get_real_travel_time_of_node_to_node(agent, action)   
        
        agent.before_time = agent.now_t
        agent.now_t += t_va_v
        agent.pi.append(action)
        agent.now_node = action

        return agent, t_va_v

    def get_path_by_dij(self, now_node, end_node):
        
        before_node = end_node
        Path = []
        while before_node != now_node:
            Path.insert(0, before_node)
            before_node = self.Predecessors[now_node][before_node]
            
        return Path

    def get_path_by_dij(self, now_node, end_node):
        
        Path = []
        next_node = now_node

        while next_node != end_node:
            Path.append(next_node)
            next_node = self.Predecessors[end_node][next_node]
    
        Path.append(end_node)

        return Path

    def Binary_insert(self, H, insert_agent):

        start = 0
        end = len(H)
        medium = end//2
        label = 0
        while(start != end):
            if insert_agent.now_t < H[medium].now_t:
                start = medium + 1
            else:
                end = medium

            medium = start + (end - start)//2

            if start >= end:
                label = 1
                break

        H.insert(start, insert_agent)

    def Binary_insery_SBP(self, insert_node, PQ, To_time, end_node):

        start = 0
        end = len(PQ)
        medium = end//2
        label = 0
        while(start != end):
            if To_time[insert_node] + self.min_travel_time[insert_node, end_node] < To_time[PQ[medium]] + self.min_travel_time[PQ[medium], end_node]:
                start = medium + 1
            else:
                end = medium

            medium = start + (end - start)//2

            if start >= end:
                label = 1
                break
        
        PQ.insert(start, insert_node)

     
    def get_distance_range_car_num(self, aim_start_node, aim_end_node, distance_low, distance_high):
        
        valid_node_list = np.where((self.shortest_distance[aim_start_node] > distance_low) & (self.shortest_distance[aim_end_node] < distance_high))[0]

        valid_street_list = []
        for valid_node in valid_node_list:
            for adjnode in self.adj_node_dic[valid_node]:
                if adjnode != valid_node and (valid_node, adjnode) not in valid_street_list:
                    if (valid_node, adjnode) in self.Street and self.shortest_distance[valid_node, aim_start_node] > self.shortest_distance[adjnode, aim_end_node]:
                        valid_street_list.append((valid_node, adjnode))
                    elif (adjnode, valid_node) in self.Street and self.shortest_distance[adjnode, aim_start_node] > self.shortest_distance[valid_node, aim_end_node]:
                        valid_street_list.append((adjnode, valid_node))
         
        flow = 0
        for valid_street in valid_street_list:
            for query in self.Street[valid_street]:
                if query.platform_class == 1 and aim_start_node in query.temp_route and aim_end_node in query.temp_route:
                    start_index = query.temp_route.index(aim_start_node)
                    end_index = query.temp_route.index(aim_end_node)
                    if start_index == end_index - 1:
                        flow += 1
         
        return flow

    def get_all_car_num(self):
        return self.car_number_of_edge.sum()

     
    def get_my_plot_car_num(self):
        return self.car_number_of_edge_my_plot.sum()
    
     
    def pre_next_time_all_car_num(self):

        y = np.array(self.history_all_car_num[-100:]).reshape(-1,1)
         
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(y)
        y = scaler.transform(y)
         
        n_lookback = 1   
        n_forecast = 1  

        X = []
        Y = []

        for i in range(n_lookback, len(y) - n_forecast + 1):
            X.append(y[i - n_lookback: i])
            Y.append(y[i: i + n_forecast])

        trainX = np.array(X)
        trainY = np.array(Y)

         
        model = Sequential()
        model.add(LSTM(4, input_shape=(n_lookback, n_forecast)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)

         
        X_ = y[- n_lookback:]   
        X_ = X_.reshape(1, n_lookback, 1)

        Y_ = model.predict(X_).reshape(-1, 1)
        Y_ = scaler.inverse_transform(Y_)[0][0]  

        return Y_

     
    def get_emergency_event_num(self):
        pass

     
    def get_use_edge(self):

        road = np.array(pd.Series(self.all_car_number_of_edge).sort_values(ascending=False)[:50].index)

        return road

    def get_dic_gap(self):

        value_gap = 0
         
        for key, value in self.searchTime.items():
            if key in self.searchTime_old:
                for s in self.searchTime[key]:
                    label_repeat = 0
                    for s_ in self.searchTime_old[key]:
                        if abs(s[0] - s_[0]) < self.same_time_gap:
                            label_repeat = 1
                            value_gap += abs(s[1] - s_[1])
                            continue
                    if label_repeat == 0:
                        value_gap += s[1]
            else:
                for s in self.searchTime[key]:
                    value_gap += s[1]
        
         
        for key, value in self.searchTime_old.items():
            if key not in self.searchTime:
                self.searchTime[key] = self.searchTime_old[key]
        
        return value_gap

    def get_d(self, r):

        return self.distance_range_gap
    
    def get_now_platform_car(self, start_node, end_node):

        flow = 0
        if (start_node, end_node) in self.Street:
            for query in self.Street[(start_node, end_node)]:
                if query.platform_class == 0:
                    flow += 1

        return flow

    def initial_Street_attr(self):
        
        street_id = 0

        for edge in self.use_edge_list:
        
            self.Street_attr[(edge[0], edge[1])] = Street_attr(0, street_id)
            street_id += 1

    def update_same_time_gap(self):

        Gap_range_list = list(np.arange(1,0,-0.005))
        for node, value in self.searchTime.items():
            if len(value) > 1:
                test = value.copy()
                gap_len_list = []
                mean_std_list = []
                Z_list = []

                for time_gap in Gap_range_list:

                    now_time = test[0][0]
                    all_group = []
                    group = [test[0][1]]
                    value_mean = []
                    value_std = []
                    for tv in test[1:]:
                        time = tv[0]
                        value = tv[1]
                        if time - now_time < time_gap:
                            group.append(value)
                        else:
                            if len(group) > 1:
                                value_mean.append(np.mean(group))
                                value_std.append(np.std(group))
                            all_group.append(group)
                            group = [value]
                            now_time = time

                    all_group.append(group)
                    value_mean.append(np.mean(group))
                    value_std.append(np.std(group))
                    if len(value_std) > 0:
                        if len(test)/len(all_group) > 1:
                            gap_len_list.append(len(all_group))
                            mean_std_list.append(np.mean(value_std))

                std_mean = np.mean(mean_std_list)
                std_std = 1 
                Z_std_list = np.array(list(map(lambda x: (x-std_mean)/std_std, mean_std_list)))

                len_mean = np.mean(gap_len_list)
                len_std = 1 
                Z_len_list = np.array(list(map(lambda x: (x-len_mean)/len_std, gap_len_list)))

                Z = np.array(0.5*Z_std_list + 0.5*Z_len_list)

                self.same_time_gap[node] = Gap_range_list[np.array(Z).argmin()]
