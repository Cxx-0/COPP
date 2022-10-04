import matplotlib.pyplot as plt
import numpy as np

REWARD_ALL = [1,2,3,4]
SEARCH_TIME_ALL = [4,5,6,7]
GTT_ALL = [6,7,8,9]

X = np.array(range(len(REWARD_ALL)))
plt.figure(1)
plt.plot(X, REWARD_ALL, label = 'REWARD')
plt.title('REWARD')
plt.figure(2)
plt.plot(X, SEARCH_TIME_ALL, label = 'SEARCH_TIME')
plt.title('REWARD')
plt.figure(3)
plt.plot(X, GTT_ALL, label = 'FAKE_REWARD')
plt.show()