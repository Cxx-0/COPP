from Street import Street_attr
from Environment import TGEnvironment
import math

# 设定agent
class envStreet():

    def __init__(self, envt: TGEnvironment):
        super(envStreet, self).__init__()
        self.envt = envt
        self.OBSERVATION_SPACE_VALUES = (2,) # 两个观察值
        self.ACTION_SPACE_VALUES = 2 # 两个动作，查询or不查询
        self.QUERY_ACTION = {'no':0, 'yes':1} # 具体查询动作
        self.INIT_QUERY_STATE = self.QUERY_ACTION['no'] # 初始不查询
        self.RETURN_IMAGE = False # 是否返回图像
    
    def reset(self):
        
        observation = []
        for street_key,street_value in self.envt.Street_dic.items():
            observation.append(self.get_observation(street_value))
        
        self.time_step = 0
        
        return observation
    
    def step(self, query_action, query_time_gap, global_travel_time, street_value, min_pre_global_travel_time):
        
        self.time_step += 1

        #observation = []
        #for street_key,street_value in self.envt.Street_dic.items():
        street_value.action(query_action, query_time_gap) # 对街道是否查询采取行动
        observation = self.get_observation(street_value)
        '''
        # 如果pre_global_travel_time比最小的大5%，则不考虑了
        if (global_travel_time - min_pre_global_travel_time)/min_pre_global_travel_time > 0.05:
            reward = np.inf
        else:
        '''
        
        #reward = math.log(self.envt.delta_time+1000)*global_travel_time # 预计通过时间作为奖励
        reward = -1 # reward在main中
        done = False
        if self.time_step > 5000:
            done = True
        
        return observation, reward, done

    def get_observation(self, street):
        
        start_node = street.Street[0]
        end_node = street.Street[1]
        '''
        # 上游下游加和
        all_car_num = 0
        for adj_node in self.envt.adj_node_dic[start_node]:
            all_car_num += self.envt.get_car_number_of_edge(start_node, adj_node)
        for adj_node in self.envt.adj_node_dic[end_node]:
            all_car_num += self.envt.get_car_number_of_edge(adj_node, end_node)
        
        # 加上当前道路值
        all_car_num += self.envt.get_car_number_of_edge(start_node, end_node)
        '''
        '''
        # 用累积车辆数作为观测值
        # 上游下游加和
        all_car_num = 0
        for adj_node in self.envt.adj_node_dic[start_node]:
            all_car_num += self.envt.all_car_number_of_edge[start_node, adj_node]
        for adj_node in self.envt.adj_node_dic[end_node]:
            all_car_num += self.envt.all_car_number_of_edge[adj_node, end_node]
        
        # 加上当前道路值
        all_car_num += self.envt.all_car_number_of_edge[start_node, end_node]
        '''

        # 当前街道的查询状态与查询间隔为obs
        obs = [street.Query_state, street.Query_time_gap]
        return obs#[all_car_num]
        
