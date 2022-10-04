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

from DDQN import DDQN
from envStreet import envStreet

def run(envt, ogp, alg):

    REWARD = []
    SEARCH_TIME = []
    FAKE_REWARD = []

    EPISODES = 1
    episode_reward = 0
    envst = envStreet(envt)
    N_ACTIONS = envst.ACTION_SPACE_VALUES
    N_STATES = envst.OBSERVATION_SPACE_VALUES[0]
    dqn = DDQN(N_STATES,N_ACTIONS)

    dqn.load_model()

    obs = [10000000000, 1]

    print(dqn.choose_action(obs,0))

    return [],[]

if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser('Setting Parameters')
    parser.add_argument('-n','--NUM_AGENTS',type=int,default=5000,help='Number of Query')
    parser.add_argument('-s','--SEED',type=int,default=20,help='random seed')
    parser.add_argument('-alg','--alg',type=int,default=2,help='choose algorithm')

    args = parser.parse_args()
    envt = TGEnvironment(args.NUM_AGENTS, args.SEED)
    ogp = OGP(envt)
    run(envt, ogp, args.alg)