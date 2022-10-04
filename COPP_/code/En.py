from Environment import Environment

from typing import List, Optional, Dict, Any

class En(object):
    """docstring for Experience"""
    envt: Optional[Environment] = None

    def __init__(self, agents: List, feasible_actions_all_agents, time: float):
        super(En, self).__init__()
