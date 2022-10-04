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
        #Load environment
        self.DATA_DIR = DATA_DIR
        self.car_speed = car_speed
        self.alpha = alpha 
        self.NUM_AGENTS = NUM_AGENTS
        self.delta_time = delta_time
        self.SEED = SEED

        self.current_time: float = 0.0
        self.last_time: float = 0.0
        self.same_time_gap = {} # 复用间隔，每个结点单独有一个复用间隔。key:结点编号, value:复用间隔
        self.initial_same_time_gap_constant = 0.001 # 初始化用
        self.distance_low_gap_percent: float = 10
        self.distance_high_gap_percent: float = 10
        self.distance_range_gap = 10 #5000*math.pi
        self.learning_rate_distance_range_gap = 2
        self.uprate_of_distance = 0.05
        self.unit_time: float = 0.08333 # 单位时间，0.08333对应5分钟
        self.action_unit_time: float = 0.08333 # 做动作的单位时间间隔，每30秒做一次动作，获取一次s,a,s_
        self.r_: float = 0.95 # 用于稀释奖励
        self.global_query_beishu = 0 # 先用全局倍数试试

        self.action_low = 0.5 # 最小倍数为0
        self.action_high = 3.5 # 最大倍数为4

        self.searchTime = {} # 用于存储上一次查询街道的时间的车辆数
        self.searchTime_old = {} # 用于存储旧查询
        self.Street = {} # 用于存储街道对应的车辆（其实就是Query）
        self.Street_attr = {} # 存储每个街道的属性
        self.history_all_car_num = [] # 用于存储历史车辆总数数据，用于预测
        self.node_attr_index = {}
        self.appear_car_num = 0 # 当前时刻新增的需要规划的车辆数

        self.dic_time_to_car = {} # 用于记录某一时刻开始的车辆的agent_id
        self.dic_time_to_global_travel_time = {} # 用于记录某一时刻开始的那批车辆的出行时间
        self.dic_time_to_query_time = {} # 用于记录某一时刻开始的那批车辆的查询次数
        self.dic_time_to_sas_ = {} # 用于记录每一个时间段内的s,a,s_，当这段时间结束后就可以补充r进行学习
        self.dic_car_number_pre = {} # 用于记录对应时间的预估车辆数，格式为key为街道，value为[[时间，车辆数]...]

        # 因为是多智能体，因此需要存储多个值
        self.temp_obs_ = [] 
        self.temp_obs = []
        self.temp_action = []
        self.temp_action_ = []

        # 强化学习参数
        self.query_gap = 0 # 当前时间段上此查的数据与当前数据的差值绝对值和
        self.query_count = 0 # 当前时间段查询的次数
        self.query_action = 0 # 下一时段是否查询的动作
        self.query_action_old = 0 # 用于保存上一次的动作

    @abstractmethod
    def initialise_environment(self):
        raise NotImplementedError

class TGEnvironment(Environment):
                                                        
    def __init__(self, NUM_AGENTS, SEED, DATA_DIR: str='../sample_data/', car_speed: int = 674, alpha: float = 0.1, delta_time: float = 0.0):  #45000
        super().__init__(DATA_DIR=DATA_DIR, car_speed=car_speed, alpha=alpha, NUM_AGENTS=NUM_AGENTS, delta_time=delta_time,SEED=SEED)

        self.initialise_environment() 

    def initialise_environment(self):
        print('Loading Environment...')
        
        SHORTESTDISTANCE_FILE: str = self.DATA_DIR + 'shortest_min_mat.npy'
        #SHORTESTDISTANCE_FILE: str = self.DATA_DIR + 'shortest.npy'

        self.shortest_distance = np.load(SHORTESTDISTANCE_FILE)

        graph = pd.read_csv(self.DATA_DIR + 'edge.csv')
        self.node_attr = pd.read_csv(self.DATA_DIR + 'node.csv')
        node_length = len(pd.Series(list(graph['start_node'])+list(graph['end_node'])).drop_duplicates())+3
        row_all = list(graph.start_node)+list(graph.end_node)
        col_all = list(graph.end_node)+list(graph.start_node)
        data_ = list(graph.distance)+list(graph.distance)
        self.TG_graph_adj = ss.coo_matrix((data_, (row_all, col_all)),shape=(node_length, node_length)).tocsr()

        print('图的结点数:', node_length)
        print('图的边数', len(graph))
        
        for i in range(self.TG_graph_adj.shape[0]):
            self.TG_graph_adj[i, i] = inf
        self.Predecessors = np.load(self.DATA_DIR+'Predecessors.npy')
        self.dontUseNode = []
        
        # ********************
        self.min_travel_time = self.shortest_distance/self.car_speed

        self.street_capacity = self.min_travel_time*45000/10 # 假设每10米一辆车

        self.mean_distance_of_edge = self.get_mean_distance_of_edge()

        self.mean_travel_time = self.mean_distance_of_edge/self.car_speed

        self.use_node = list(range(self.TG_graph_adj.shape[0]))

        data = []
        row = []
        col = []
        self.node_length = self.TG_graph_adj.shape[0]
        self.car_number_of_edge = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()
        self.car_number_of_edge_my_plot = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil() # 当前平台车辆数
        self.pre_car_number_of_edge = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()
        self.ganzhi_car_number_of_edge = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()
        self.all_car_number_of_edge = {} #ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()
        self.car_num_avg = {} # key：街道 value: 车辆数list
        
        self.other_car_number_of_edge = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()

        self.index_array = np.array(range(self.TG_graph_adj.shape[0])) 

        self.adj_node_dic = self.initial_adj_node_dic()
        np.save(self.DATA_DIR+'adj_node_dic.npy', self.adj_node_dic)
        # *********************

        self.L =  [[[] for i in range(self.node_length)] for i in range(self.node_length)]
        #self.L_other = [[[] for i in range(self.node_length)] for i in range(self.node_length)]

        # 不加tolist表示利用查询到的边对应时刻的flow表示相邻边对应时刻的flow
        self.use_edge = np.load(self.DATA_DIR+'use_edge.npy', allow_pickle=True)[:50]
        '''
        # 使用所有边
        self.use_edge = []
        for r,c in zip(row_all,col_all):
            self.use_edge.append([r,c])
        self.use_edge = np.array(self.use_edge)
        '''
        self.use_edge_list = self.use_edge.tolist()  
        print('查询边个数:', len(self.use_edge))
        self.use_node_H = []

        self.lowest_distance = 100
        self.highest_distance = 2000

        self.initial_Street_attr() # 初始化道路属性
        self.initial_node_attr_index() # 初始化结点以及其对应坐标
        self.initial_centrality() # 初始化结点中心度
        self.initial_same_time_gap() # 初始化复用间隔

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
        self.distance_range_gap = 10 #5000*math.pi
        self.learning_rate_distance_range_gap = 0.05
        self.uprate_of_distance = 0.05
        self.unit_time: float = 0.08333 # 单位时间，0.08333对应5分钟
        self.action_unit_time: float = 0.08333 # 做动作的单位时间间隔，每30秒做一次动作，获取一次s,a,s_
        self.r_: float = 0.95 # 用于稀释奖励
        self.global_query_beishu = 0 # 先用全局倍数试试

        self.action_low = 0.5 # 最小倍数为0
        self.action_high = 3.5 # 最大倍数为4

        self.searchTime = {} # 用于存储上一次查询街道的时间的车辆数
        self.searchTime_old = {} # 用于存储旧查询
        self.Street = {} # 用于存储街道对应的车辆（其实就是Query）
        self.Street_attr = {} # 存储每个街道的属性
        self.history_all_car_num = [] # 用于存储历史车辆总数数据，用于预测
        self.node_attr_index = {}
        self.appear_car_num = 0 # 当前时刻新增的需要规划的车辆数

        self.dic_time_to_car = {} # 用于记录某一时刻开始的车辆的agent_id
        self.dic_time_to_global_travel_time = {} # 用于记录某一时刻开始的那批车辆的出行时间
        self.dic_time_to_query_time = {} # 用于记录某一时刻开始的那批车辆的查询次数
        self.dic_time_to_sas_ = {} # 用于记录每一个时间段内的s,a,s_，当这段时间结束后就可以补充r进行学习
        self.dic_car_number_pre = {} # 用于记录对应时间的预估车辆数，格式为key为街道，value为[[时间，车辆数]...]

        # 因为是多智能体，因此需要存储多个值
        self.temp_obs_ = [] 
        self.temp_obs = []
        self.temp_action = []
        self.temp_action_ = []

        # 强化学习参数
        self.query_gap = 0 # 当前时间段上此查的数据与当前数据的差值绝对值和
        self.query_count = 0 # 当前时间段查询的次数
        self.query_action = 0 # 下一时段是否查询的动作
        self.query_action_old = 0 # 用于保存上一次的动作

        self.initial_Street_attr() # 初始化道路属性
        self.initial_node_attr_index() # 初始化结点以及其对应坐标
        self.initial_centrality() # 初始化结点中心度

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

        #  first_node_ = 500 # 88 1023 500
        
        first_node = [first_node_]
        self.use_node_1 = [first_node_]
        '''
        count = 0
        while count < 300: #len(self.use_node):
            now_use_node = first_node.pop()
            for new_node in self.adj_node_dic[now_use_node]:
                if new_node not in self.use_node_1: #and self.shortest_distance[now_use_node, new_node] < 150:
                    count += 1
                    first_node.insert(0, new_node)
                    if len(self.get_adj_node(new_node)) > 2:
                        self.use_node_1.insert(0, new_node)
        '''
        #np.save(self.DATA_DIR+'use_node_.npy',self.use_node_1)
        #self.use_node_1 = random.sample(self.use_node[:-2], 250) #随机挑选N个点作为重要点
        
        self.use_node_1 = list(range(0,200))
        dis = 0
        for i in range(NUM_AGENTS):
            a = random.sample(self.use_node_1, 1)[0]
            b = random.sample(self.use_node_1, 1)[0]   
            
            while (a == b or self.shortest_distance[a][b] < self.lowest_distance or self.shortest_distance[a][b] > self.highest_distance): #避免起点终点相同, 以及限制出行距离
                b = random.sample(self.use_node_1, 1)[0]
                if self.shortest_distance[a,b] == np.inf:
                    a = random.sample(self.use_node_1, 1)[0]

            dis += self.shortest_distance[a,b]
            initial_query.append([a, b])
        
        '''    
        for i in range(NUM_AGENTS):
            a = random.sample(self.use_node, 1)[0]
            b = random.sample(self.use_node, 1)[0]   

            while (a == b or self.shortest_distance[a, b] < self.lowest_distance or self.shortest_distance[a, b] > self.highest_distance): #避免起点终点相同, 以及限制出行距离
                b = random.sample(self.use_node, 1)[0]

            if a not in self.use_node_H:
                self.use_node_H.append(a)
            if b not in self.use_node_H:
                self.use_node_H.append(b)

            initial_query.append([a, b])    
        '''  
        np.random.seed(self.SEED)
        query_time = np.array(list(np.random.normal(6.3,0.1,int(NUM_AGENTS/2))) + list(np.random.normal(6.8,0.1,int(NUM_AGENTS/2))))
        #query_time = [random.uniform(6.1, 6.5) for i in range(NUM_AGENTS)] # 晚一点出现，先让其余的跑一段时间，使车辆数稳定
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

            while (a == b or self.shortest_distance[a][b] < self.lowest_distance or self.shortest_distance[a][b] > self.highest_distance): #避免起点终点相同, 以及限制出行距离
                b = random.sample(self.use_node_1, 1)[0]
                if self.shortest_distance[a,b] == np.inf:
                    a = random.sample(self.use_node_1, 1)[0]
                
            initial_query.append([a, b])   
            
        np.random.seed(self.SEED+20) 
        query_time = np.array(list(np.random.normal(6.3,0.1,int(NUM_AGENTS/2))) + list(np.random.normal(6.8,0.1,int(NUM_AGENTS/2))))
        #query_time = [random.uniform(6,6.5) for i in range(NUM_AGENTS)]
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
        mean_distance_of_edge = 15.874 #TG 15.874429663999985   chengdu 1051 
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
        
        #C = 20
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
        self.car_number_of_edge_my_plot = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil() # 当前平台车辆数
        self.pre_car_number_of_edge = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()
        self.ganzhi_car_number_of_edge = ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()
        self.all_car_number_of_edge = {} #ss.coo_matrix((data, (row, col)),shape=(self.node_length, self.node_length)).tolil()
        self.car_num_avg = {} # key：街道 value: 车辆数list

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

    # 获取距离目标节点在一定距离范围内，且通过目标道路的车辆
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
        #start_time = time.time()
        flow = 0
        for valid_street in valid_street_list:
            for query in self.Street[valid_street]:
                if query.platform_class == 1 and aim_start_node in query.temp_route and aim_end_node in query.temp_route:
                    start_index = query.temp_route.index(aim_start_node)
                    end_index = query.temp_route.index(aim_end_node)
                    if start_index == end_index - 1:
                        flow += 1
        #print(time.time() - start_time, len(valid_street_list))
        return flow

    '''
    获取观察值
    '''
    # 获取当前道路车辆总数
    def get_all_car_num(self):
        return self.car_number_of_edge.sum()

    # 获取当前平台车辆总数
    def get_my_plot_car_num(self):
        return self.car_number_of_edge_my_plot.sum()
    
    # 预测下一时刻道路车辆总数
    def pre_next_time_all_car_num(self):

        '''
        利用LSTM预测
        '''

        # 建立预测序列
        y = np.array(self.history_all_car_num[-100:]).reshape(-1,1)
        # scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(y)
        y = scaler.transform(y)
        # generate the input and output sequences
        n_lookback = 1  # length of input sequences (lookback period)
        n_forecast = 1 # length of output sequences (forecast period)

        X = []
        Y = []

        for i in range(n_lookback, len(y) - n_forecast + 1):
            X.append(y[i - n_lookback: i])
            Y.append(y[i: i + n_forecast])

        trainX = np.array(X)
        trainY = np.array(Y)

        # LSTM
        model = Sequential()
        model.add(LSTM(4, input_shape=(n_lookback, n_forecast)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)

        # generate the forecasts
        X_ = y[- n_lookback:]  # last available input sequence
        X_ = X_.reshape(1, n_lookback, 1)

        Y_ = model.predict(X_).reshape(-1, 1)
        Y_ = scaler.inverse_transform(Y_)[0][0] # 预测下一时刻车辆数

        return Y_

    # 当前道路紧急事件数
    def get_emergency_event_num(self):
        pass

    # 获取需要被查询的街道
    def get_use_edge(self):

        road = np.array(pd.Series(self.all_car_number_of_edge).sort_values(ascending=False)[:50].index)

        return road

    def get_dic_gap(self):

        value_gap = 0
        # 遍历新查询，如果新查询有而旧查询没有，则加全值，否则两值相减取绝对值
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
        
        # 遍历旧查询，如果新查询没有，则补全
        for key, value in self.searchTime_old.items():
            if key not in self.searchTime:
                self.searchTime[key] = self.searchTime_old[key]
        
        return value_gap

    def get_d(self, r):

        return self.distance_range_gap
        '''
        if r < 1:
            return np.sqrt(self.distance_range_gap/math.pi)
        else:
            return self.distance_range_gap/(4*math.pi*r)
        '''
    
    # 获取街道上当前平台的车辆数
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
                std_std = 1#np.std(mean_std_list)
                Z_std_list = np.array(list(map(lambda x: (x-std_mean)/std_std, mean_std_list)))

                len_mean = np.mean(gap_len_list)
                len_std = 1#np.std(gap_len_list)
                Z_len_list = np.array(list(map(lambda x: (x-len_mean)/len_std, gap_len_list)))

                Z = np.array(0.5*Z_std_list + 0.5*Z_len_list)

                self.same_time_gap[node] = Gap_range_list[np.array(Z).argmin()]
