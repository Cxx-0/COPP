class Query():

    def __init__(self, start_node, end_node, now_node, now_t, pi, agent_id, need_plan_list, before_time, platform_class):
        self.start_node = start_node
        self.end_node = end_node
        self.now_node = now_node
        self.now_t = now_t
        self.pi = pi
        self.agent_id = agent_id  
        self.temp_route = []
        self.pred_arrive_time = now_t
        self.appear_time = now_t
        self.need_plan_list = need_plan_list
        self.response_time = []
        self.before_time = before_time
        self.platform_class = platform_class  
        self.query_table = {}  
        self.belong_start_time = 0 # 用于记录属于那个开始时间段的