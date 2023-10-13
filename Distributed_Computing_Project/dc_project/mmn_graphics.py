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
choices = [1,2,5,10] 
#choices = [15,20,50,100]
        
#labels for the x axis
x_labels = [2,4,6,8,10,12,14]
#labels for the y axis
y_labels = [0.0,0.2,0.4,0.6,0.8,1.0]
#line style
style = ['solid','dashed','dashdot','dotted']

# create figure for subplots
figure, axis = plt.subplots(2, 2)
figure.tight_layout(pad = 3.0)

#perform the four graphs simoultaneasly
gx = 0
gy = 0
# every choice of d has a different graph 
for c in choices:
    if c == 2:
        gx = 0
        gy = 1
    elif c == 5:
        gx = 1
        gy = 0
    elif c == 10:
        gx = 1
        gy = 1
    else:
        gx = 0
        gy = 0
    # counter for the linestyle of the different lambda graphs
    counter = 0  
    # with the 4 iteration for the different lambda
    for l in lambd:
        sim = mmn_queue.MMN(l, 1, n, c, 1)
        sim.run(1_000) # default 1_000_000
        completions = sim.completions
        # output of completions
        W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
        print(f"Average time spent in the system: {W} \n with lambda : {l} and choices : {c}")
        #print(f"Theoretical expectation for random server choice: {1 / (1 - l)}")
        
        # throw out the final_array
        final_arr = []
        norm_final_arr = []
        indexes = [i for i in range(0,15)]
        for i in indexes:
            final_arr.append(sum((x >= i) for x in sim.plot))

        # normalizing
        norm_final_arr = [float(i)/max(final_arr) for i in final_arr]

        # plotting         
        axis[gx, gy].plot(indexes[1:], norm_final_arr[1:], label=f"lambda : {l}", linestyle=style[counter]) 
        counter = counter + 1
        axis[gx, gy].set_title(f"{c}"+" choices")
        axis[gx,gy].set_xlabel("Queue lenght")
        axis[gx,gy].set_ylabel("Fraction of queues with at least that size")
    # plot for each choices    
    axis[gx, gy].legend(loc=0, prop={'size': 7})
    axis[gx, gy].grid()
    axis[gx, gy].set_xticks(x_labels)
    axis[gx, gy].set_yticks(y_labels)

plt.show()