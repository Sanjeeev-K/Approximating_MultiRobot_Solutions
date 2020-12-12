import random as rnd
import math
import numpy as np
import os
import sys
import json
from TO_SysPathGenerator import SysPathGenerator


class Instance_Generator:

    def __init__(self, no_of_robots, no_of_depots, no_of_tasks, delta, Tmax,
                 no_of_instances=1, seed_list=[rnd.randrange(sys.maxsize)],
                 path_to_data_folder=os.getcwd()):
        self.noOfRobots = no_of_robots
        self.noOfTasks = no_of_tasks
        self.noOfDepots = no_of_depots
        self.L = 1
        self.T_max = Tmax
        self.vel = 1
        self.arenaRadius = 100
        self.delta = delta
        self.no_of_instances = no_of_instances
        if len(seed_list) == no_of_instances:
            self.seed_list = seed_list
        elif len(seed_list) == 1:
        # Handling the case if only no_of_instances are provided
            self.seed_list = []
            for i in range(self.no_of_instances):
                self.seed_list.append(rnd.randrange(sys.maxsize))
        else:
            raise Exception('Number of instances not equal to number of seeds.')
        self.filePaths = SysPathGenerator(self.noOfRobots, self.noOfDepots, self.noOfTasks, 
                                            self.delta, self.T_max)
        self.instance_folder_path = self.filePaths.instance_data_folder_path
        self.instance_filename_prefix = self.filePaths.instance_data_filename_prefix

    # Function for generating a random location in a circle
    def random_xy_loc_in_circle(self):
        r=self.arenaRadius*rnd.random()
        theta = 2*math.pi*rnd.random()
        x=r*np.cos(theta)
        y=r*np.sin(theta)
        return x,y

    def create_instance(self, iteration, seed):
        self.iteration = iteration
        self.thisSeed = seed
        rnd.seed(self.thisSeed)
        self.K = ["K" + str(i) for i in range(self.noOfRobots)]
        self.T = ["T" + str(i) for i in range(self.noOfTasks)]
        self.D = ["D" + str(i) for i in range(self.noOfDepots)]
        self.S = ['S0']
        self.E = ['E0']
        self.N = self.T + self.D + self.S + self.E

        # R = {task: 1 for task in T}

        self.T_loc = {task: (self.random_xy_loc_in_circle()) for task in self.T}

        # for tasks equally distributed in a circle
        # for i in range (self.noOfTasks):
            # self.T_loc["T" + str(i)]= (0.5*self.arenaRadius*math.cos(2*math.pi*(i/(self.noOfTasks))),0.5*self.arenaRadius*math.sin(2*math.pi*(i/(self.noOfTasks))))
            # self.T_loc["T" + str(i)]= (0.5*self.arenaRadius*math.cos(2*math.pi*(2*i/(self.noOfTasks)))+0.1*i,0.1*i + 0.5*self.arenaRadius*math.sin(2*math.pi*(2*i/(self.noOfTasks))))
            # self.T_loc["T" + str(i)]= (0.5*self.arenaRadius,0.5*self.arenaRadius)

        # self.sum_of_all_task_distances_from_origin = 0.0
        # for i in range (self.noOfTasks):
        #     x = self.T_loc[i][0]
        #     y = self.T_loc[i][1]
        #     self.sum_of_all_task_distances_from_origin += math.sqrt (x**2 + y**2)
        # self.mean_task_distance = self.sum_of_all_task_distances_from_origin/self.noOfTasks
        # self.noOfRobots = int (self.mean_task_distance / self.delta )

        
        # Set Depot locations based on number of robots.
        # If |D| = 1 : Place it at the center
        # If |D| = 2 : Place them at locations (0, R/2) and (0, -R/2)
        # If |D| = 3 : Place them at locations (0, sqrt(3)*delta/8), 
        #          : (-delta/4, -sqrt(3)*delta/8), (delta/4, -sqrt(3)*delta/8)
        # If |D| = 4 : Place them at locations (delta/4, delta/4)
        #          : (delta/4, -delta/4), (-delta/4, delta/4)
        #          : (-delta/4, -delta/4)

        if self.noOfDepots == 1:
            self.D_loc = {loc: (0,0) for loc in self.D}
        elif self.noOfDepots == 2:
            self.D_loc = {self.D[0]: (0, self.arenaRadius/2), 
                          self.D[1]: (0, -self.arenaRadius/2)}
        elif self.noOfDepots == 3:
            self.D_loc = {self.D[0]: (0, math.sqrt(3)*self.delta/8), 
                          self.D[1]: (-self.delta/4, -math.sqrt(3)*self.delta/8),
                          self.D[2]: (self.delta/4, -math.sqrt(3)*self.delta/8)}
        elif self.noOfDepots == 4:
            self.D_loc = {self.D[0]: (self.delta/4, self.delta/4), 
                          self.D[1]: (-self.delta/4, self.delta/4),
                          self.D[2]: (self.delta/4, -self.delta/4),
                          self.D[3]: (-self.delta/4, -self.delta/4)}

        
        self.S_loc = {loc: self.D_loc['D0'] for loc in self.S}
        self.E_loc = {loc: self.D_loc['D0'] for loc in self.E}
        self.N_loc = {**self.T_loc, **self.D_loc, **self.S_loc, **self.E_loc}

        self.edges = [(i, j) for i in self.N for j in self.N if i != j]
        self.c = {t: np.linalg.norm(
            np.array(self.N_loc.get(t[0]))-np.array(self.N_loc.get(t[1])))
            for t in iter(self.edges)}
        self.f = {t: (1/self.delta)*np.linalg.norm(
            np.array(self.N_loc.get(t[0]))-np.array(self.N_loc.get(t[1]))) 
            for t in iter(self.edges)}


        self.arcs = [(i, j, k)
                     for i in self.N for j in self.N for k in self.K if i != j]
        self.arc_ub = {
            (i, j, k): 1 for i in self.N for j in self.N for k in self.K if i != j}
        for arc in self.arc_ub:
            if arc[0] in self.D and arc[1] in self.D:
                self.arc_ub[arc] = self.noOfTasks
        self.k_y = [(i, k) for i in self.T for k in self.K]

    def create_json_data(self):
        self.json_data = {
            'iteration': self.iteration,
            'thisSeed': self.thisSeed,
            'noOfTasks': self.noOfTasks,
            'noOfDepots': self.noOfDepots,
            'noOfRobots': self.noOfRobots,
            'arenaRadius': self.arenaRadius,
            'K': self.K,
            'T': self.T,
            'D': self.D,
            'S': self.S,
            'E': self.E,
            'N': self.N,

            'L': self.L,
            'delta': self.delta,
            'vel': self.vel,
            'T_max': self.T_max,
            # 'R': self.R,

            'T_loc': self.T_loc,
            'D_loc': self.D_loc,
            'S_loc': self.S_loc,
            'E_loc': self.E_loc,
            'N_loc': self.N_loc,

            'edges': self.edges,
            'c': dict((':'.join(k), v) for k, v in self.c.items()),
            'f': dict((':'.join(k), v) for k, v in self.f.items()),

            'arcs': [[arc[0], arc[1], arc[2]] for arc in self.arcs],
            'k_y': self.k_y,
            'arc_ub': dict((':'.join(k), v) for k, v in self.arc_ub.items())
        }

    def write_instance_to_json(self):
        curr_instance_filename = self.instance_filename_prefix + \
            'Iter' + str(self.iteration) + '.json'
        file_path = os.path.normpath(
            self.instance_folder_path+curr_instance_filename)

        if not os.path.exists(self.instance_folder_path):
            os.makedirs(self.instance_folder_path)

        with open(file_path, 'w') as fp:
            json.dump(self.json_data, fp, sort_keys=True, indent=4)

    def generate_data(self):
        for iteration, seed in zip(range(self.no_of_instances), self.seed_list):
            self.create_instance(iteration, seed)
            self.create_json_data()
            self.write_instance_to_json()


def main():
    min_robots = 1
    max_robots = 10

    min_depots = 1
    max_depots = 3

    min_tasks = 2 # not useful
    max_tasks = 10 # not useful

    delta_range_start = 150
    delta_range_step = 100
    # delta_range_end = int(math.ceil(2*100*math.sqrt(2) /
    #                               delta_range_step)*delta_range_step)  # ~282
    delta_range_end = 150

    Tmax_range_start = 600
    Tmax_range_step = 100
    # Tmax_range_end = int(math.ceil(2*100*math.sqrt(2) /
    #                               Tmax_range_step)*Tmax_range_step)  # ~282
    Tmax_range_end = 600

    robots_range = list(range(min_robots, max_robots+1))
    depots_range = list(range(min_depots, max_depots+1))
    #tasks_range = list(range(min_tasks, max_tasks+1))
    tasks_range = [25,30,35,40]                                                        # change here.
    #delta_range = list(range(delta_range_start, delta_range_end +
    #                        delta_range_step, delta_range_step))
    delta_range = [300, 600]
    #Tmax_range = list(range(Tmax_range_start, Tmax_range_end +
    #                        Tmax_range_step, Tmax_range_step,))
    Tmax_range = [50, 75, 150, 300, 450, 600]

    no_of_instances = 10
    #seed_list = [rnd.randrange(sys.maxsize) for i in range(no_of_instances)]
    seed_list = [1679156726667891581,
                1650489432282747979,
                3433114415903346549,
                5237005969715870065,
                3503955511357013226,
                4756079252721248023,
                2281456846044426304,
                3390157364023920803,
                8762155956704656379,
                5406991502605904493]

    for r in robots_range:
        for d in depots_range:
            for t in tasks_range:
                for f in delta_range:
                    for tmax in Tmax_range:
                        instance = Instance_Generator(
                            r, d, t, f, tmax, no_of_instances, seed_list)
                        instance.generate_data()
                        print('Data generated for: R{}D{}T{}Delta{}Tmax{}'.format(r,d,t,f,tmax))


if __name__ == "__main__":
    main()
