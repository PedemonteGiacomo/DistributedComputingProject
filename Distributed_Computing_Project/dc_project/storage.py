# ALI HAIDER S4811831
# GIACOMO PEDEMONTE S4861715

#!/usr/bin/env python
import argparse
import configparser
import logging
import random
from dataclasses import dataclass
from random import expovariate
from typing import Optional, List
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import plotly.express as px

# compile with 
# python3 .\storage.py p2p.cfg --verbose --max-t 200years --seed 7

# the humanfriendly library (https://humanfriendly.readthedocs.io/en/latest/) lets us pass parameters in human-readable
# format (e.g., "500 KiB" or "5 days"). You can safely remove this if you don't want to install it on your system, but
# then you'll need to handle sizes in bytes and time spans in seconds--or write your own alternative.
# It should be trivial to install (e.g., aptg install python3-humanfriendly or conda/pip install humanfriendly).
from humanfriendly import format_timespan, parse_size, parse_timespan

from discrete_event_sim import Simulation, Event

count_backup = 0
count_successed_restore = 0
count_failed_restore = 0

def exp_rv(mean):
    """Return an exponential random variable with the given mean."""
    return expovariate(1 / mean)

class DataLost(Exception):
    """Not enough redundancy in the system, data is lost. We raise this exception to stop the simulation."""
    pass

class Backup(Simulation):
    """Backup simulation."""

    # type annotations for `Node` are strings here to allow a forward declaration:
    # https://stackoverflow.com/questions/36193540/self-reference-or-forward-reference-of-type-annotations-in-python
    def __init__(self, nodes: List['Node']):
        super().__init__()  # call the __init__ method of parent class
        self.nodes = nodes

        # we add to the event queue the first event of each node going online and of failing
        for node in nodes:
            self.schedule(node.arrival_time, Online(node))
            self.schedule(node.arrival_time + exp_rv(node.average_lifetime), Fail(node))

    def schedule_transfer(self, uploader: 'Node', downloader: 'Node', block_id: int, restore: bool):
        """Helper function called by `Node.schedule_next_upload` and `Node.schedule_next_download`.

        If `restore` is true, we are restoring a block owned by the downloader, otherwise, we are saving one owned by
        the uploader.
        """
        
        block_size = downloader.block_size if restore else uploader.block_size

        assert uploader.current_upload is None
        assert downloader.current_download is None
        
        # if the downloader is an evil nodes refus any kind of download
        if downloader.evil and restore is False and IF_peer and SELFISH_nodes:
            self.log_info(f"Refused download from {downloader}")
            return

        speed = min(uploader.upload_speed, downloader.download_speed)  # we take the slowest between the two
        delay = block_size / speed
        if restore:
            event = BlockRestoreComplete(uploader, downloader, block_id)
        else:
            event = BlockBackupComplete(uploader, downloader, block_id)
        self.schedule(delay, event)
        uploader.current_upload = downloader.current_download = event
        
        #was commented
        self.log_info(f"scheduled {event.__class__.__name__} from {uploader} to {downloader}"
                       f" in {format_timespan(delay)}")

    def log_info(self, msg):
        """Override method to get human-friendly logging for time."""

        logging.info(f'{format_timespan(self.t)}: {msg}')

@dataclass(eq=False)  # auto initialization from parameters below (won't consider two nodes with same state as equal)
class Node:
    """Class representing the configuration of a given node."""

    # using dataclass is (for our purposes) equivalent to having something like
    # def __init__(self, description, n, k, ...):
    #     self.n = n
    #     self.k = k
    #     ...
    #     self.__post_init__()  # if the method exists

    name: str  # the node's name

    n: int  # number of blocks in which the data is encoded
    k: int  # number of blocks sufficient to recover the whole node's data

    data_size: int  # amount of data to back up (in bytes)
    storage_size: int  # storage space devoted to storing remote data (in bytes)

    upload_speed: float  # node's upload speed, in bytes per second
    download_speed: float  # download speed

    average_uptime: float  # average time spent online
    average_downtime: float  # average time spent offline

    average_lifetime: float  # average time before a crash and data loss
    average_recover_time: float  # average time after a data loss

    arrival_time: float  # time at which the node will come online
    
    evil: bool # evil behavior of the node

    def __post_init__(self):
        """Compute other data dependent on config values and set up initial state."""

        # whether this node is online. All nodes start offline.
        self.online: bool = False

        # whether this node is currently under repairs. All nodes are ok at start.
        self.failed: bool = False

        # size of each block
        self.block_size: int = self.data_size // self.k if self.k > 0 else 0

        # amount of free space for others' data 
        self.free_space: int = self.storage_size - self.block_size * self.n
        #note we always leave enough space for our n blocks
        assert self.free_space >= 0, "Node without enough space to hold its own data"
        # removed the equal to get the 

        # local_blocks[block_id] is true if we locally have the local block
        # [x] * n is a list with n references to the object x
        self.local_blocks: list[bool] = [True] * self.n

        # backed_up_blocks[block_id] is the peer we're storing that block on, or None if it's not backed up yet;
        # we start with no blocks backed up
        self.backed_up_blocks: list[Optional[Node]] = [None] * self.n

        # (owner -> block_id) mapping for remote blocks stored
        self.remote_blocks_held: dict[Node, int] = {}

        # current uploads and downloads, stored as a reference to the relative TransferComplete event
        self.current_upload: Optional[TransferComplete] = None
        self.current_download: Optional[TransferComplete] = None
        
        # self.score: int = 0 # score to obtain a sort of behavioral knowledge of the nodes

    def find_block_to_back_up(self):
        """Returns the block id of a block that needs backing up, or None if there are none."""

        # find a block that we have locally but not remotely
        # check `enumerate` and `zip`at https://docs.python.org/3/library/functions.html
        for block_id, (held_locally, peer) in enumerate(zip(self.local_blocks, self.backed_up_blocks)):
            # (0, (local_block[0],backed_up_blocks[0]) ; 1, (local_block[1],backed_up_blocks[1]) ; ......)
            if held_locally and peer is None:
                return block_id
        return None

    def schedule_next_upload(self, sim: Backup):
        """Schedule the next upload, if any."""

        assert self.online

        if self.current_upload is not None:
            return

        # first find if we have a backup that a remote node needs
        for peer, block_id in self.remote_blocks_held.items():
            # if the block is not present locally and the peer is online and not downloading anything currently, then
            # schedule the restore from self to peer of block_id 
            if peer.online and peer.current_download is None and not peer.local_blocks[block_id]: #and peer in self.remote_blocks_held
                sim.schedule_transfer(uploader=self, downloader=peer, block_id=block_id, restore=True)
                return  # we have found our upload, we stop

        # try to back up a block on a locally held remote node
        block_id = self.find_block_to_back_up()
        if block_id is None:
            return
        #era commentato
        sim.log_info(f"{self} is looking for somebody to back up block {block_id}")
        remote_owners = set(node for node in self.backed_up_blocks if node is not None)  # nodes having one block
        for peer in sim.nodes:
            # if the peer is not self, is online, is not among the remote owners = (and peer not in remote_owners), has enough space and is not
            # downloading anything currently, schedule the backup of block_id from self to peer only if self helds a block of the peer
            if (peer is not self and peer.online and peer not in remote_owners and peer.current_download is None
                    and peer.free_space >= self.block_size):
                # implementing the Tit For Tat for the p2p system: 
                # we do the upload only if the peer that wants to download have done uploads 
                # on the uploader node(so if in the uploader.remote blocks is present the downloader)
                if (peer in self.remote_blocks_held) and IF_peer: # alternative at the "Optimistic Unchoke" : starting the tit for tat behaviour after x years(or sim.t < parse_timespan("20 Years"))
                    sim.schedule_transfer(uploader=self, downloader=peer, block_id=block_id, restore=False)
                    return
                # to make more clients behaviors correctly with the servers and doesn't store anything in the client nodes
                if "client" not in peer.name and not IF_peer: 
                    sim.schedule_transfer(uploader=self, downloader=peer, block_id=block_id, restore=False)
                    return
                # decomment this in order to obtain that clients can store data (comment the previous)
                #if not IF_peer:
                #    sim.schedule_transfer(uploader=self, downloader=peer, block_id=block_id, restore=False)
                #    return
        # if we go here maybe we choke some peer that are not behaving like self.
        # selection of random nodes to obtain the "Optimistic Unchoke" 
        if IF_peer and sim.t: # > parse_timespan("20 Years")
            rand_peer = random.choice(sim.nodes)
            if (rand_peer is not self and rand_peer.online and rand_peer not in remote_owners and rand_peer.current_download is None
                        and rand_peer.free_space >= self.block_size):
                sim.schedule_transfer(uploader=self, downloader=rand_peer, block_id=block_id, restore=False)

    def schedule_next_download(self, sim: Backup):
        """Schedule the next download, if any."""

        assert self.online
        
        if self.current_download is not None:
            #sim.log_info(f"{self}: current downloading")
            return

        # first find if we have a missing block to restore
        for block_id, (held_locally, peer) in enumerate(zip(self.local_blocks, self.backed_up_blocks)):
            if not held_locally and peer is not None and peer.online and peer.current_upload is None: #and peer in self.remote_blocks_held:
                sim.schedule_transfer(uploader=peer, downloader=self, block_id=block_id, restore=True)
                #sim.log_info(f"schedule_next_download on {self}")
                return  # we are done in this case

        # try to back up a block for a remote node
        for peer in sim.nodes:
            if (peer is not self and peer.online and peer.current_upload is None and peer not in self.remote_blocks_held
                    and self.free_space >= peer.block_size): # and self in peer.remote_blocks_held
                block_id = peer.find_block_to_back_up()
                if block_id is not None:
                    # checking if I'm backuping a block for the node whose storing almost a block for me
                    # if self have done upload on peer, peer unchoke the upload and all goes on.
                    if (self in peer.remote_blocks_held) and IF_peer : # or sim.t < parse_timespan("50 Years")
                        sim.schedule_transfer(uploader=peer, downloader=self, block_id=block_id, restore=False)
                        #sim.log_info(f"schedule_next_download on {self}")
                        return
                    # for the client server configuration we make the download of a block only if we are not client nodes
                    if "client" not in self.name and not IF_peer:
                        sim.schedule_transfer(uploader=peer, downloader=self, block_id=block_id, restore=False)
                        #sim.log_info(f"schedule_next_download on {self}")
                        return
                    # decomment this in order to obtain that clients can store data (comment the previous)
                    #if not IF_peer:
                    #    sim.schedule_transfer(uploader=peer, downloader=self, block_id=block_id, restore=False)
                    #    return
        # if we go here maybe we choke some peer that are not behaving like self.
        # selection of random nodes to obtain the "Optimistic Unchoke" 
        if IF_peer:
            rand_peer = random.choice(sim.nodes)
            if (rand_peer is not self and rand_peer.online and rand_peer.current_upload is None and rand_peer not in self.remote_blocks_held
                        and self.free_space >= rand_peer.block_size):
                block_id = rand_peer.find_block_to_back_up()
                if block_id is not None:
                        sim.schedule_transfer(uploader=rand_peer, downloader=self, block_id=block_id, restore=False)
                        #sim.log_info(f"schedule_next_download on {self}")
                        return
                
    def __hash__(self):
        """Function that allows us to have `Node`s as dictionary keys or set items.

        With this implementation, each node is only equal to itself.
        """
        return id(self)

    def __str__(self):
        """Function that will be called when converting this to a string (e.g., when logging or printing)."""

        return self.name  

@dataclass
class NodeEvent(Event):
    """An event regarding a node. Carries the identifier, i.e., the node's index in `Backup.nodes_config`"""

    node: Node

    def process(self, sim: Simulation):
        """Must be implemented by subclasses."""
        raise NotImplementedError


class Online(NodeEvent):
    """A node goes online."""

    def process(self, sim: Backup):
        node = self.node
        if node.online or node.failed:
            return
        node.online = True
        # schedule next upload and download
        node.schedule_next_upload(sim)
        node.schedule_next_download(sim)
        # schedule the next offline event
        # for our vision this means that the node goes offline after the average uptime(before without the exp_rv)
        sim.schedule(exp_rv(node.average_uptime), Offline(node))


class Recover(Online):
    """A node goes online after recovering from a failure."""

    def process(self, sim: Backup):
        node = self.node
        sim.log_info(f"{node} recovers")
        node.failed = False
        super().process(sim)
        sim.schedule(exp_rv(node.average_lifetime), Fail(node))
        node.free_space = node.storage_size - node.block_size * node.n

class Disconnection(NodeEvent):
    """Base class for both Offline and Fail, events that make a node disconnect."""

    def process(self, sim: Simulation):
        """Must be implemented by subclasses."""
        raise NotImplementedError

    def disconnect(self):
        node = self.node
        node.online = False
        # cancel current upload and download
        # retrieve the nodes we're uploading and downloading to and set their current downloads and uploads to None
        current_upload, current_download = node.current_upload, node.current_download
        if current_upload is not None:
            current_upload.canceled = True
            current_upload.downloader.current_download = None
            node.current_upload = None
        if current_download is not None:
            current_download.canceled = True
            current_download.uploader.current_upload = None
            node.current_download = None


class Offline(Disconnection):
    """A node goes offline."""

    def process(self, sim: Backup):
        node = self.node
        if node.failed or not node.online:
            return
        assert node.online
        self.disconnect()
        # schedule the next online event
        sim.schedule(exp_rv(self.node.average_downtime), Online(node))


class Fail(Disconnection):
    """A node fails and loses all local data."""

    def process(self, sim: Backup):
        sim.log_info(f"{self.node} fails")
        self.disconnect()
        node = self.node
        node.failed = True
        node.local_blocks = [False] * node.n  # lose all local data
        # lose all remote data
        for owner, block_id in node.remote_blocks_held.items():
            owner.backed_up_blocks[block_id] = None
            if owner.online and owner.current_upload is None:
                owner.schedule_next_upload(sim)  # this node may want to back up the missing block
        node.remote_blocks_held.clear()
        # schedule the next online and recover events
        sim.schedule(exp_rv(node.average_recover_time), Recover(node))


@dataclass
class TransferComplete(Event):
    """An upload is completed."""

    uploader: Node
    downloader: Node
    block_id: int
    canceled: bool = False

    def __post_init__(self):
        assert self.uploader is not self.downloader

    def process(self, sim: Backup):
        sim.log_info(f"{self.__class__.__name__} from {self.uploader} to {self.downloader}")
        if self.canceled:
            return  # this transfer was canceled, so ignore this event
        uploader, downloader = self.uploader, self.downloader
        assert uploader.online and downloader.online
        self.update_block_state()
        uploader.current_upload = downloader.current_download = None
        uploader.schedule_next_upload(sim)
        downloader.schedule_next_download(sim)
        for node in [uploader, downloader]:
            sim.log_info(f"{node}: {sum(node.local_blocks)} local blocks, "
                         f"{sum(peer is not None for peer in node.backed_up_blocks)} backed up blocks, "
                         f"{len(node.remote_blocks_held)} remote blocks held")

    def update_block_state(self):
        """Needs to be specified by the subclasses, `BlockBackupComplete` and `BlockRestoreComplete`."""
        raise NotImplementedError


class BlockBackupComplete(TransferComplete):

    def update_block_state(self):
        global count_backup
        owner, peer = self.uploader, self.downloader
        peer.free_space -= owner.block_size
        assert peer.free_space >= 0
        owner.backed_up_blocks[self.block_id] = peer
        peer.remote_blocks_held[owner] = self.block_id
        count_backup += 1


class BlockRestoreComplete(TransferComplete):
    
    def update_block_state(self):
        global count_successed_restore
        global count_failed_restore
        owner = self.downloader
        owner.local_blocks[self.block_id] = True
        if sum(owner.local_blocks) == owner.k:  # we have exactly k local blocks, we have all of them then
            owner.local_blocks = [True] * owner.n
            count_successed_restore += 1
        else:
            count_failed_restore += 1

def LostBlocks(sim: Backup):
    nodes = sim.nodes
    l_blocks = 0
    t_blocks = 0
    for node in nodes:
        for i in range(node.n):
            t_blocks += 1
            if node.local_blocks[i] == False and node.backed_up_blocks[i] is None:
                l_blocks += 1       
    print(f"lost blocks = {l_blocks} of {t_blocks}\n")
    return l_blocks

def main(config_file, number_of_evil): 
    parser = argparse.ArgumentParser()
    #parser.add_argument("config", help="configuration file")
    parser.add_argument("--max-t", default="100 years")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        # set a seed to make experiments repeatable
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')  # output info on stdout

    # functions to parse every parameter of peer configuration
    parsing_functions = [
        ('n', int), ('k', int),
        ('data_size', parse_size), ('storage_size', parse_size),
        ('upload_speed', parse_size), ('download_speed', parse_size),
        ('average_uptime', parse_timespan), ('average_downtime', parse_timespan),
        ('average_lifetime', parse_timespan), ('average_recover_time', parse_timespan),
        ('arrival_time', parse_timespan) , ('evil', bool)
    ]
    
    config = configparser.ConfigParser()
    config.read(config_file)
    nodes = []  
    # we build the list of nodes to pass to the Backup class
    for node_class in config.sections():
        class_config = config[node_class]
        # list comprehension: https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions
        cfg = [parse(class_config[name]) for name, parse in parsing_functions]
        # the `callable(p1, p2, *args)` idiom is equivalent to `callable(p1, p2, args[0], args[1], ...)
        nodes.extend(Node(f"{node_class}-{i}", *cfg) for i in range(class_config.getint('number')))
        #appends to the empty list the view of all the nodes, client and server side
        #print(class_config.getboolean('evil'))
    sim = Backup(nodes)
    
    if SELFISH_nodes:
    # set the number of evil nodes 
        for node in sim.nodes[number_of_evil:]:
            node.evil = False
            
    sim.run(parse_timespan(args.max_t))
    sim.log_info(f"Simulation over")
    
    # final print in case of not considering selfish nodes
    for node in sim.nodes:
        remote_peer = []
        if IF_peer:
            remote_peer = []
            for peer, block_id in node.remote_blocks_held.items():
                remote_peer.append(peer.name)
        print(f"{node}: {sum(node.local_blocks)} local blocks, "
                        f"{sum(peer is not None for peer in node.backed_up_blocks)} backed up blocks, "
                        f"{len(node.remote_blocks_held)} remote blocks held,\n"
                        f"list of remote blocks held {remote_peer}") # print to obtain the items inside the node.remote_blocks_held in order to see who are the owners of blocks that I'm backupping in my space for those owners
        if SELFISH_nodes:
            print(f"{node.evil}")
    return LostBlocks(sim=sim)

# set the config type in order to distinguish client server runs from peer to peer
IF_peer = True 
SELFISH_nodes = False
# number of test in order to repeat simulation to obtain different results from the main
n_test = 10
# considering the casse of the peer-to peer system
if IF_peer is True: 
    configurations = ["p2p.cfg"] # base configuration, comment this line and decomment the other configurations array in order to prove with all the configurations file reported in the /config directory
    x_ticks = ["p2p_evil"] # change also the ticks of the plot with the correct ones
    if __name__ == '__main__':
        configs_losts = []
        if SELFISH_nodes: # considering increasing number of selfish nodes
            #configurations = ["config/balanced.cfg","config/cost_efficient.cfg","config/high_availability_2.cfg","config/low_data_loss.cfg"]
            #x_ticks = ["balanced","cost_efficient","high_availability_2","low_data_loss"]
            for config in configurations:
                x = []
                configs_losts = []
                for j in range(0,15):
                    x.append(j)
                    lost = []
                    for i in range(n_test):
                        print(f"Simulation number {i+1} of the configuration in : {config}")
                        lost.append(main(config,number_of_evil=j))
                    configs_losts.append(np.mean(lost))
                #plt.bar(configs_losts, label=config)
                plt.plot(x, configs_losts, label=config)
                plt.xlabel('Evil')
                plt.ylabel('Lost blocks with different evils')
                plt.title('Line Plot of Selfish nodes impact')
            plt.legend()
            plt.show() 
        else:
            # different types of configurations in order to accomplish different needs if wants to consider all of those decomment the following lines
            #configurations = ["config/balanced.cfg","config/cost_efficient.cfg","config/high_availability.cfg","config/high_availability_2.cfg","config/high_data_econding.cfg","config/high_storage.cfg","config/low_cost.cfg","config/low_latency.cfg","config/low_resource.cfg","config/low_data_loss.cfg"]
            configurations = ["config2/config1.cfg","config2/config2.cfg","config2/config3.cfg","config2/config4.cfg","config2/config5.cfg","config2/config6.cfg"]
            #configurations = ["config3/balanced.cfg","config3/cost_efficient.cfg","config3/high_availability.cfg","config3/high_availability_2.cfg","config3/high_data_econding.cfg","config3/high_storage.cfg","config3/low_cost.cfg","config3/low_latency.cfg","config3/low_resource.cfg","config3/low_data_loss.cfg"]
            #x_ticks = ["balanced","cost_efficient","high_availability","high_availability_2","high_data_econding","high_storage","low_cost","low_latency","low_resource","low_data_loss"]
            x_ticks = [4,5,6,7,8,9]
            
            for config in configurations:
                    lost = []
                    for i in range(n_test):
                        print(f"Simulation number {i+1} of the configuration in : {config}")
                        lost.append(main(config,0))
                    configs_losts.append((lost))
            # plot
            plt.figure(figsize=(12, 7))
            plt.title("Blocks losts during different simulations for each configuration")
            plt.xlabel("Configurations")
            plt.ylabel("Data Losts for configurations")
            ax = sns.boxplot(data=configs_losts,palette=["m", "g", "r", "b"] )
            ax.set_xticklabels(x_ticks)
            plt.setp(ax.get_xticklabels(), rotation=30)
            plt.show()
# Otherwise run the client server configuration
else: 
    configurations = ["client_server.cfg"]
    x_ticks = ["client_server"]
    if __name__ == '__main__':
        configs_losts = []
        for config in configurations:
            lost = []
            for i in range(n_test):
                print(f"Simulation number {i+1} of the configuration in : {config}")
                lost.append(main(config,0))
            configs_losts.append((lost))
        # plot
        plt.figure(figsize=(12, 7))
        plt.title("Blocks losts during different simulations for each configuration")
        plt.xlabel("Configurations")
        plt.ylabel("Data Losts for configurations")
        ax = sns.boxplot(data=configs_losts,palette=["m", "g", "r", "b"] )
        ax.set_xticklabels(x_ticks)
        plt.setp(ax.get_xticklabels(), rotation=30)
        plt.show()
        