 
class Street_attr:
    def __init__(self, Query_state, Street_id):  
         
        self.Query_state = Query_state  
         
        self.Street_id = Street_id  
        self.Query_time_gap = 0
        self.query_beishu = 0  
        
    
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