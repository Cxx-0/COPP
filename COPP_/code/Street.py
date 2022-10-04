# 定义街道实体
class Street_attr:
    def __init__(self, Query_state, Street_id): # centrality
        # 街道状态
        self.Query_state = Query_state # 当前街道是否查询
        #self.centrality = centrality # 道路中心性
        self.Street_id = Street_id # 道路编号
        self.Query_time_gap = 0
        self.query_beishu = 0 # 查询倍数
        
    
    def action(self, choice_query, choice_time_gap):
        if choice_query == 0:
            self.Query_state = 0
        elif choice_query == 1:
            self.Query_state = 1

        if choice_time_gap == 0:
            self.Query_time_gap = 0
        elif choice_time_gap == 1:
            self.Query_time_gap = 1
        elif choice_time_gap == 2:
            self.Query_time_gap = 2
        elif choice_time_gap == 3:
            self.Query_time_gap = 3
        elif choice_time_gap == 4:
            self.Query_time_gap = 4