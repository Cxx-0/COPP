from Street import Street_attr
from Environment import TGEnvironment
import math

 
class envStreet():

    def __init__(self, envt: TGEnvironment):
        super(envStreet, self).__init__()
        self.envt = envt
        self.OBSERVATION_SPACE_VALUES = (2,)  
        self.ACTION_SPACE_VALUES = 2  
        self.QUERY_ACTION = {'no':0, 'yes':1}  
        self.INIT_QUERY_STATE = self.QUERY_ACTION['no']  
        self.RETURN_IMAGE = False  
    
    def reset(self):
        
        observation = []
        for street_key,street_value in self.envt.Street_dic.items():
            observation.append(self.get_observation(street_value))
        
        self.time_step = 0
        
        return observation
    
    def step(self, query_action, query_time_gap, global_travel_time, street_value, min_pre_global_travel_time):
        
        self.time_step += 1

         
         
        street_value.action(query_action, query_time_gap)  
        observation = self.get_observation(street_value)

        reward = -1  
        done = False
        if self.time_step > 5000:
            done = True
        
        return observation, reward, done

    def get_observation(self, street):
        
        start_node = street.Street[0]
        end_node = street.Street[1]
        
        obs = [street.Query_state, street.Query_time_gap]
        return obs 
        
