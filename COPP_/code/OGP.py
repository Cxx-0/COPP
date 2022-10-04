from cmath import inf
from En import En
from Environment import TGEnvironment
from typing import List, Tuple, Deque, Dict, Any, Iterable
import numpy as np
import time

class OGP():

    def __init__(self, envt: TGEnvironment):
        super(OGP, self).__init__()
        self.envt = envt
    
    def Inf(self, agents):
        final_action= []
        for agent in agents:
            if agent.now_node == agent.start_node:       
                adj_to_end_path = self.envt.get_path_by_dij(agent.now_node, agent.end_node)
                agent.temp_route = adj_to_end_path[1:]
                now_node = agent.start_node
                count = 0
                while now_node != agent.end_node:

                    adj_node = agent.temp_route[count]
                    flow = self.envt.get_agent_number_of_edge(now_node, adj_node)

                    min_travel_time = self.envt.get_min_travel_time(now_node, adj_node)
                    t_va_v = self.envt.get_edge_travel_time_by_edge_density_nonlinear(min_travel_time, flow)

                     

                    agent.pred_arrive_time +=  t_va_v
                    now_node = adj_node
                    count += 1

                final_action.append(agent.temp_route.pop(0))
            else:    
                final_action.append(agent.temp_route.pop(0))

        return final_action

    def InitSearch(self, agent, remove_edge):        
        pi = []
        arrive_time = []
        flow_num = []
        PQ = [agent.now_node]
        pre_node = np.zeros(self.envt.node_length)
        To_time = np.zeros(self.envt.node_length)
        Flow_num = np.zeros(self.envt.node_length)
        To_time[To_time==0] = 99999999
        To_time[agent.now_node] = agent.now_t

        while len(PQ) != 0:
            v = PQ.pop()
            if v == agent.end_node:
                while v != agent.now_node:
                    v = int(v)
                    pi.insert(0, v)
                    arrive_time.insert(0, To_time[v])
                    flow_num.insert(0, Flow_num[v])
                    self.envt.L[int(pre_node[v])][v].append((To_time[int(pre_node[v])],To_time[v]))
                    v = pre_node[v]   
                break

            for adj_node in self.envt.get_adj_node(v):

                flow = 0
                
                for time_label in self.envt.L[v][adj_node]:    
                    ta = time_label[0]
                    tb = time_label[1]
                     
                    if ta <= To_time[v] and tb >= To_time[v]:
                        flow += 1
                
                 
                if v != remove_edge[0] and adj_node != remove_edge[1] and v != adj_node:
                    
                    if [v, adj_node] in self.envt.use_edge:
                        
                        use_v = -1
                        use_adj_node = -1
                        dont_query_state1 = 0 
                         
                        if v not in self.envt.searchTime and adj_node not in self.envt.searchTime:
                            
                             

                             
                            if [v, adj_node] in self.envt.use_edge_list and self.envt.Street_attr[(v, adj_node)].Query_state == 1:
                                use_v = v
                                use_adj_node = adj_node

                             
                            elif [v, adj_node] in self.envt.use_edge:
                                for adjnode in self.envt.adj_node_dic[v]:
                                     
                                    if [adjnode, v] in self.envt.use_edge_list and self.envt.Street_attr[(adjnode, v)].Query_state == 1:
                                        use_v = adjnode
                                        use_adj_node = v
                                        break
                                     
                                    if [v, adjnode] in self.envt.use_edge_list and self.envt.Street_attr[(v, adjnode)].Query_state == 1:
                                        use_v = v
                                        use_adj_node = adjnode
                                        break
                                if use_v == -1:
                                    for adjnode in self.envt.adj_node_dic[adj_node]:
                                         
                                        if [adj_node, adjnode] in self.envt.use_edge_list and self.envt.Street_attr[(adj_node, adjnode)].Query_state == 1:
                                            use_v = adj_node
                                            use_adj_node = adjnode
                                            break
                                         
                                        if [adjnode, adj_node] in self.envt.use_edge_list and self.envt.Street_attr[(adjnode, adj_node)].Query_state == 1:
                                            use_v = adjnode
                                            use_adj_node = adj_node
                                            break
                            
                             
                            if use_v != -1:

                                self.envt.delta_time += 1

                                 
                                d = self.envt.get_d((To_time[v] - agent.now_t) * self.envt.car_speed)
                                distance_low = (To_time[v] - agent.now_t) * self.envt.car_speed-d 
                                distance_high = (To_time[v] - agent.now_t) * self.envt.car_speed+d 
                                if distance_low < 0:
                                    distance_low = 0
                                other_flow = self.envt.get_distance_range_car_num(use_v, use_adj_node, distance_low, distance_high)

                                if (use_v, use_adj_node) not in self.envt.dic_car_number_pre:
                                    self.envt.dic_car_number_pre[(use_v, use_adj_node)] = [[To_time[v], other_flow]]
                                else:
                                    self.envt.dic_car_number_pre[(use_v, use_adj_node)].append([To_time[v], other_flow])

                                flow += other_flow  

                                if use_v in self.envt.searchTime and To_time[v] < 900000:
                                    self.envt.searchTime[use_v].append([To_time[v] - agent.now_t, other_flow])
                                elif To_time[v] < 900000:
                                    self.envt.searchTime[use_v] = [[To_time[v] - agent.now_t, other_flow]]

                                if use_adj_node in self.envt.searchTime and To_time[v] < 900000:
                                    self.envt.searchTime[use_adj_node].append([To_time[v] - agent.now_t, other_flow])
                                elif To_time[v] < 900000:
                                    self.envt.searchTime[use_adj_node] = [[To_time[v] - agent.now_t, other_flow]]

                            else:
                                dont_include_label = 1

                         
                        else:
                            
                             
                            label = 0
                             
                            if v in self.envt.searchTime:
                                for time_gap, car_num in self.envt.searchTime[v]:
                                    if abs(time_gap - (To_time[v] - agent.now_t)) < self.envt.same_time_gap[v]:
                                        flow += car_num
                                        label = 1
                                        break
                            if label == 0 and adj_node in self.envt.searchTime:
                                for time_gap, car_num in self.envt.searchTime[adj_node]:
                                    if abs(time_gap - (To_time[adj_node] - agent.now_t)) < self.envt.same_time_gap[adj_node]:
                                        flow += car_num
                                        label = 1
                                        break

                            if [v, adj_node] in self.envt.use_edge_list and self.envt.Street_attr[(v, adj_node)].Query_state == 1:
                                use_v = v
                                use_adj_node = adj_node
                            elif [v, adj_node] in self.envt.use_edge:
                                for adjnode in self.envt.adj_node_dic[v]:
                                     
                                    if [adjnode, v] in self.envt.use_edge_list and self.envt.Street_attr[(adjnode, v)].Query_state == 1:
                                        use_v = adjnode
                                        use_adj_node = v
                                        break
                                     
                                    if [v, adjnode] in self.envt.use_edge_list and self.envt.Street_attr[(v, adjnode)].Query_state == 1:
                                        use_v = v
                                        use_adj_node = adjnode
                                        break
                                if use_v == -1:
                                    for adjnode in self.envt.adj_node_dic[adj_node]:
                                         
                                        if [adj_node, adjnode] in self.envt.use_edge_list and self.envt.Street_attr[(adj_node, adjnode)].Query_state == 1:
                                            use_v = adj_node
                                            use_adj_node = adjnode
                                            break
                                         
                                        if [adjnode, adj_node] in self.envt.use_edge_list and self.envt.Street_attr[(adjnode, adj_node)].Query_state == 1:
                                            use_v = adjnode
                                            use_adj_node = adj_node
                                            break

                             
                            if label == 0 and use_v != -1 and use_adj_node != -1:
                                
                                self.envt.delta_time += 1

                                d = self.envt.get_d((To_time[v] - agent.now_t) * self.envt.car_speed)
                                distance_low = (To_time[v] - agent.now_t) * self.envt.car_speed-d 
                                distance_high = (To_time[v] - agent.now_t) * self.envt.car_speed+d 
                                if distance_low < 0:
                                    distance_low = 0
                                other_flow = self.envt.get_distance_range_car_num(use_v, use_adj_node, distance_low, distance_high)
                                
                                if (use_v, use_adj_node) not in self.envt.dic_car_number_pre:
                                    self.envt.dic_car_number_pre[(use_v, use_adj_node)] = [[To_time[v], other_flow]]
                                else:
                                    for c_index, c_value in enumerate(self.envt.dic_car_number_pre[(use_v, use_adj_node)]):
                                        if To_time[v] < c_value[0]:
                                            self.envt.dic_car_number_pre[(use_v, use_adj_node)].insert(c_index, [To_time[v], other_flow])
                                            break
                                flow += other_flow  

                                if use_v in self.envt.searchTime and To_time[v] < 900000:
                                    self.envt.searchTime[use_v].append([To_time[v] - agent.now_t, other_flow])
                                elif To_time[v] < 900000:
                                    self.envt.searchTime[use_v] = [[To_time[v] - agent.now_t, other_flow]]

                                if use_adj_node in self.envt.searchTime and To_time[v] < 900000:
                                    self.envt.searchTime[use_adj_node].append([To_time[v] - agent.now_t, other_flow])
                                elif To_time[v] < 900000:
                                    self.envt.searchTime[use_adj_node] = [[To_time[v] - agent.now_t, other_flow]]
                
                flow += self.envt.car_number_of_edge[v, adj_node]

                min_travel_time = self.envt.get_min_travel_time(v, adj_node)

                T = self.envt.get_edge_travel_time_by_edge_density_nonlinear(min_travel_time, flow)   

                if To_time[v] + T < To_time[adj_node]:

                    To_time[adj_node] = To_time[v] + T
                    pre_node[adj_node] = v
                    Flow_num[adj_node] = flow
                    if adj_node not in PQ:
                        self.envt.Binary_insery_SBP(adj_node, PQ, To_time, agent.end_node)

        return pi, To_time[agent.end_node] - To_time[agent.start_node]
    
    def SBP(self, agents, remove_edge):

        final_action = []

        not_init_agent_index = []

        pre_global_travel_time = 0

        for agent_index, agent in enumerate(agents):
            
            if len(agent.temp_route) == 0:

                pi, one_pre_global_travel_time = self.InitSearch(agent, remove_edge)

                pre_global_travel_time += one_pre_global_travel_time

                not_init_agent_index.append(agent_index)

                pi.reverse()
                agent.temp_route = pi
                
                final_action.append(agent.temp_route.pop())

            else:
                final_action.append(agent.temp_route.pop())

        return final_action, pre_global_travel_time

    def InitSearch_other(self, agent):        
        pi = []
        arrive_time = []
        flow_num = []
        PQ = [agent.now_node]
        pre_node = np.zeros(self.envt.node_length)
        To_time = np.zeros(self.envt.node_length)
        Flow_num = np.zeros(self.envt.node_length)
        To_time[To_time==0] = 99999999
        To_time[agent.now_node] = agent.now_t

        while len(PQ) != 0:
            v = PQ.pop()
            if v == agent.end_node:
                while v != agent.now_node:
                    v = int(v)
                    pi.insert(0, v)
                    arrive_time.insert(0, To_time[v])
                    flow_num.insert(0, Flow_num[v])
                     
                    v = pre_node[v]   
                break
                
            for adj_node in self.envt.get_adj_node(v):
            
                flow = 0
                
                flow += self.envt.car_number_of_edge[v, adj_node]

                '''
                for time_label in self.envt.L_other[v][adj_node]:   
                    ta = time_label[0]
                    tb = time_label[1]
                    if ta <= To_time[v] and tb >= To_time[v]:
                        flow += 1    
                '''
                min_travel_time = self.envt.get_min_travel_time(v, adj_node)

                T = self.envt.get_edge_travel_time_by_edge_density_nonlinear(min_travel_time, flow)   

                if To_time[v] + T < To_time[adj_node]:

                    To_time[adj_node] = To_time[v] + T
                    pre_node[adj_node] = v
                    Flow_num[adj_node] = flow
                    if adj_node not in PQ:
                        self.envt.Binary_insery_SBP(adj_node, PQ, To_time, agent.end_node)

        return pi

    def SBP_other(self, agents):

        final_action = []

        not_init_agent_index = []

        for agent_index, agent in enumerate(agents):
            
            if len(agent.temp_route) == 0:

                pi = self.InitSearch_other(agent)

                not_init_agent_index.append(agent_index)

                pi.reverse()
                agent.temp_route = pi

                final_action.append(agent.temp_route.pop())

            else:
                final_action.append(agent.temp_route.pop())

        return final_action

    def Greedy(self, agents):

        final_action = []
        
        for agent in agents:
            if len(agent.temp_route) == 0:
                now_node = agent.start_node 
                while now_node != agent.end_node:

                    EGT_min = inf
                    one_agent_final_action = -1
                    t_va_v_min = inf

                    for adj_node in self.envt.get_adj_node(now_node):
                        if adj_node not in agent.pi and self.envt.get_min_travel_time(adj_node, agent.end_node) < self.envt.get_min_travel_time(now_node, agent.end_node):

                            flow = self.envt.get_agent_number_of_edge(now_node, adj_node)

                            min_travel_time = self.envt.get_min_travel_time(now_node, adj_node)
                            t_va_v = self.envt.get_edge_travel_time_by_edge_density_nonlinear(min_travel_time, flow)
                            EGT = t_va_v + self.envt.get_min_travel_time(adj_node, agent.end_node)

                            if EGT < EGT_min:
                                one_agent_final_action = adj_node
                                EGT_min = EGT
                                t_va_v_min = t_va_v

                     
                    now_node = one_agent_final_action
                    agent.temp_route.append(now_node)
                    agent.pred_arrive_time = agent.pred_arrive_time+t_va_v_min

                agent.temp_route.reverse()

                final_action.append(agent.temp_route.pop())
            else:
                final_action.append(agent.temp_route.pop())

        return final_action

    def InitSearch_dynamic(self, agent):        

        all_pi = []
        all_time = []
        all_adj = []

        best_time = 99999999

        count = -1
        for adj_node_first in self.envt.get_adj_node(agent.now_node):
            
            
            if adj_node_first not in agent.pi:
            
                count += 1
                pi = []
                arrive_time = []
                flow_num = []

                pre_node = np.zeros(self.envt.node_length)
                To_time = np.zeros(self.envt.node_length)

                PQ = [adj_node_first]
                To_time[To_time==0] = 99999999

                min_travel_time = self.envt.get_min_travel_time(agent.now_node, adj_node_first)
                flow1 = self.envt.car_number_of_edge[agent.now_node, adj_node_first] 
                T1 = self.envt.get_edge_travel_time_by_edge_density_nonlinear(min_travel_time, flow1)
                To_time[adj_node_first] = agent.now_t + T1

                while len(PQ) != 0:
                    v = PQ.pop()
                    if v == agent.end_node:
                        while v != adj_node_first:
                            v = int(v)
                            pi.insert(0, v)
                            v = pre_node[v]  
                        break
                        
                    for adj_node in self.envt.get_adj_node(v):
                        
                        if adj_node not in agent.pi:

                            flow = 0
                            flow += self.envt.car_number_of_edge[v, adj_node] 

                            min_travel_time = self.envt.get_min_travel_time(v, adj_node)
                            T = self.envt.get_edge_travel_time_by_edge_density_nonlinear(min_travel_time, flow)
                            if To_time[v] + T < To_time[adj_node]:
                                To_time[adj_node] = To_time[v] + T
                                pre_node[adj_node] = v
                                if adj_node not in PQ:
                                    self.envt.Binary_insery_SBP(adj_node, PQ, To_time, agent.end_node)

                all_pi.append(pi)
                all_time.append(To_time[agent.end_node])
                all_adj.append(adj_node_first)

                if To_time[agent.end_node] < best_time:
                    pi.insert(0,adj_node_first)
                    arrive_time.append(T1)
                    flow_num.append(flow1)
                    best_time = To_time[agent.end_node]
                    select = count

        return select, all_pi, all_time, all_adj

    def Greedy_OGP(self, agents):

        final_action = []

        all_one_select = []
        all_all_pi = []
        all_all_time = []
        all_all_adj = []

        for agent_index, agent in enumerate(agents):
            
            one_select, one_all_pi, one_all_time, one_all_adj = self.InitSearch_dynamic(agent)
            all_all_adj.append(one_all_adj)
            all_one_select.append(one_select)
            all_all_pi.append(one_all_pi)
            all_all_time.append(one_all_time)
        
        now_node = agent.now_node
        dicFlow = {}

        for adj_node in self.envt.get_adj_node(now_node):
            dicFlow[adj_node] = 0

        for agent_index, agent in enumerate(agents):
            adj_one = all_all_adj[agent_index][all_one_select[agent_index]]
            dicFlow[adj_one] += 1
    
        for agent_index, agent in enumerate(agents):  
            pi = all_all_pi[agent_index][all_one_select[agent_index]]
            pi.reverse()
            agent.temp_route = pi

            final_action.append(agent.temp_route.pop())


        return final_action
            

    