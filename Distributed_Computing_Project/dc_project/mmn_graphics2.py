#!/usr/bin/env python3
import mmn_queue
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import math

#data
n = 100
lambd = [0.5,0.9,0.95,0.99]
choice = [1, 2]
#sim = MMN(l, 1, 1000, choice, 1)
#sim.run(1_000_000)
#print(lambd)

#number of tests
n_test = 100       
#labels for the x axis
x_labels = [2,4,6,8,10,12,14]
#labels for the y axis
y_labels = [0.0,0.2,0.4,0.6,0.8,1.0]
#line style
style = ['solid','dashed','dashdot','dotted']
# with the 4 iteration for the different lambda
for c in choice:
    data_W = []
    for l in lambd:
        W_list = []
        for t in range(n_test):
            sim = mmn_queue.MMN(l, 1, n, c, 1)
            sim.run(1_000_000) # default 1_000_000
            completions = sim.completions
            #print(completions.values())
            #print(sim.arrivals)
            # output of completions
            W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
            print(f"Average time spent in the system: {W} \n with lambda : {l} and choices : {c}")
            #print(f"Theoretical expectation for random server choice: {1 / (1 - l)}")
            W_list.append(W)
        data_W.append(np.mean(W_list))
    plt.plot(lambd, data_W, label=f"{c} choices")
    #plt.plot(lambd, data_TW, label='Theoretical expectation for random server choice')
plt.xlabel('Lambda')
plt.ylabel('Average Time in System')
plt.title('Line Plot of Average Time in System and Theoretical Expectation')
plt.legend()
plt.show()
            
    
