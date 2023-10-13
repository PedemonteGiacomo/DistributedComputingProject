# ALI HAIDER S4811831
# GIACOMO PEDEMONTE S4861715

#!/usr/bin/env python
import argparse
import csv
import collections
import math
import random
from random import expovariate
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from discrete_event_sim import Simulation, Event

class MMN(Simulation):

    def __init__(self, lambd, mu, n, d, plot):
        if n < 1:
            raise NotImplementedError  # extend this to make it work for multiple queues

        super().__init__()
        self.running = [ None for i in range(n) ] #None
        self.queue = [ collections.deque() for i in range(n) ] #collections.deque()#[ collections.deque() for i in range(n) ]  # FIFO queue of the system
        self.queue_length = [ None for i in range(n) ] #make an array of queue length that represent at some time of the exce
        self.arrivals = {}  # dictionary mapping job id to arrival time
        self.completions = {}  # dictionary mapping job id to completion time
        self.lambd = lambd
        self.n = n
        self.mu = mu
        self.arrival_rate = lambd*n #/n in the previous version
        self.completion_rate = mu #/n in the previous version
        # save the information about the supermarket choice
        self.d = d
        self.plot_interval = plot
        self.plot = []
        self.schedule(expovariate(lambd), Arrival(0))
        self.schedule(0, MonitorMMN(plot))

    def schedule_arrival(self, job_id):
        # schedule the arrival following an exponential distribution, 
        # to compensate the number of queues the arrival
        # time should depend also on "n"
        self.schedule(expovariate(self.arrival_rate), event=Arrival(job_id=job_id))
        # ogni tot fai il monitor

    def schedule_completion(self, job_id):
        # schedule the time of the completion event
        self.schedule(expovariate(self.completion_rate), event=Completion(job_id=job_id))

    #@property
    def queue_len(self, i):
        return (self.running[i] is not None) + len(self.queue[i])     
        #in the multiqueues scenario change like this

class Arrival(Event):
    
    def __init__(self, job_id):
        self.id = job_id
    
    def process(self, sim: MMN): 
        # set the arrival time of the job
        sim.arrivals.update({self.id:sim.t})
        # find the queue with the minimum size (SUPERMARKET)
        d_queues_indexes = random.sample(range(sim.n), sim.d)
        in_running = False
        min_length = math.inf
        final_index = 0
        # with the supermarket we need to choose the first queue that the running is none. 
        for i in d_queues_indexes:
            q_lenght = sim.queue_len(i)
            if sim.queue_len(i) == 0:
                sim.running[i] = self.id
                sim.schedule_completion(job_id=sim.running[i])
                in_running = True
                break
            if q_lenght < min_length:
                min_length = q_lenght
                final_index = i
        if not in_running:
            sim.queue[final_index].append(self.id)
        # schedule the arrival of the next job
        sim.schedule_arrival(job_id=self.id+1)

class Completion(Event):
    
    def __init__(self, job_id):
        self.id = job_id 

    def process(self, sim: MMN):
        i = sim.running.index(self.id)
        #check if the index exists
        assert i is not None
        # set the completion time of the running job
        sim.completions.update({self.id:sim.t})
        # if the queue is not empty
        if sim.queue[i]:
            # get a job from the queue
            job = sim.queue[i].popleft()
            # schedule its completion
            sim.running[i] = job
            sim.schedule_completion(job_id=job)
        else:
            #if the queue of an index is empty, no running job are associated to that index
            sim.running[i] = None

# sampling            
class MonitorMMN(Event):
    """At any configurable interval, we save the lenght of the queues."""
    
    def __init__(self, interval=1):
        self.interval = interval

    def process(self, sim: MMN):
        # needs to count the queue lenght in this "monitor moment", save in a file with lambd and d
        for i in range(sim.n):
            #sim.queue_length[i]=len(sim.queue[i])
            # if we consider the sim.queue_len(i) we'll obtain a different vision considering the running state of the queues
            # So, the graph goes down from 2 instead from 1 in this version
            sim.plot.append(sim.queue_len(i)) 
            # save all the samples in an array of lenght 
        sim.schedule(self.interval, self)
    
def main(lambd):
    # parsing
    parser = argparse.ArgumentParser()
    #parser.add_argument('--lambd', type=float, default=0.7)
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--max-t', type=float, default=1_000)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument("--plot_interval", type=float, default=1, help="how often to collect data points for the plot")
    args = parser.parse_args()
    
    # needs to choose a right number of d that is minor than n
    assert args.n >= args.d 

    # initiate and run the simultaion
    sim = MMN(lambd, args.mu, args.n, args.d, args.plot_interval)
    sim.run(args.max_t)

    # output for completions
    completions = sim.completions
    W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
    print(f"Average time spent in the system: {W}")
    T_W = 1 / (1 - lambd)
    print(f"Theoretical expectation for random server choice: {T_W}")     

    # csv file
    if args.csv is not None:
        with open(args.csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([lambd, args.mu, args.max_t, W, args.d, args.n])
    return W,T_W

n_test = 10
lambd = [0.5] #, 0.7, 0.9, 0.95, 0.99]
#lambd = [2,4,8,16,32]
data_W = []
data_TW = []
if __name__ == '__main__':
    for l in lambd:
        W_list = []
        TW_list = []
        for i in range(n_test):
            print(i)
            W, TW = main(l)
            W_list.append(W)
            TW_list.append(TW)
        avg_of_avg_time_in_system = np.mean(W_list)
        avg_of_TW = np.mean(TW_list)
        data_W.append(avg_of_avg_time_in_system)
        data_TW.append(avg_of_TW)

    plt.plot(lambd, data_W, label='Average time spent in the system')
    plt.plot(lambd, data_TW, label='Theoretical expectation for random server choice')
    plt.xlabel('Lambda')
    plt.ylabel('Average Time in System')
    plt.title('Line Plot of Average Time in System and Theoretical Expectation')
    plt.legend()
    plt.show()
