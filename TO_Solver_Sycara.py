from gurobipy import *
import random as rnd
import math
import numpy as np
import os
from TO_InstanceReader import InstanceReader


class Solver_Sycara:

    def __init__(self, instance):
        '''
        param instance: A single instance that can be solved by the MinMax Solver
        '''
        self.iteration = instance.iteration
        self.noOfRobots = instance.noOfRobots
        self.noOfTasks = instance.noOfTasks
        self.noOfDepots = instance.noOfDepots
        self.L = instance.L
        self.T_max = instance.T_max
        self.vel = instance.vel
        self.thisSeed = instance.thisSeed
        self.K = instance.K
        self.T = instance.T
        self.D = instance.D
        self.S = instance.S
        self.E = instance.E
        self.N = instance.N
        self.L = instance.L
        self.vel = instance.vel
        self.T_max = instance.T_max
        #R = json_data['R']
        self.T_loc = instance.T_loc
        self.D_loc = instance.D_loc
        self.S_loc = instance.S_loc
        self.E_loc = instance.E_loc
        self.N_loc = instance.N_loc
        self.edges = instance.edges
        self.c = instance.c
        self.f = instance.f
        self.arcs = instance.arcs
        self.arc_ub = instance.arc_ub
        self.k_y = instance.k_y
        # Generate names for model/solution_file
        self.path_to_sol_folder = os.getcwd()
        self.instance_folder_path_suffix = \
            '/sol' + \
            '/R' + str(self.noOfRobots) + \
            '/D' + str(self.noOfDepots) + \
            '/T' + str(self.noOfTasks) + \
            '/F' + str(self.L) + \
            '/Tmax' + str(self.T_max)
        self.instance_folder_path = os.path.normpath(
            self.path_to_sol_folder + self.instance_folder_path_suffix)
        self.instance_filename_prefix = '/R' + str(self.noOfRobots) + \
            'D' + str(self.noOfDepots) + \
            'T' + str(self.noOfTasks) + \
            'F' + str(self.L) + \
            'Tmax' + str(self.T_max)
        self.curr_instance_filename = self.instance_filename_prefix + \
            'Iter' + str(self.iteration)
        self.file_path = os.path.normpath(
            self.instance_folder_path + self.curr_instance_filename)

        # Initialize the model
        self.init_model()

    def init_model(self):
        # Initialize the model
        self.model = Model(
            'D_Mitchel_et_al_2015-' + self.curr_instance_filename[1:] +
            '-Seed:' + str(self.thisSeed))
        # Decision variables and their bounds
        x = model.addVars(arcs, lb=0, ub=arc_ub, name="x", vtype=GRB.INTEGER)
        p = model.addVars(arcs, name="p", vtype=GRB.INTEGER)
        P_max = model.addVar(name="P_max")
        r = model.addVars(T, lb=0, ub=L, vtype=GRB.CONTINUOUS, name="r")

        # Objective function
        model.setObjective(P_max, GRB.MINIMIZE)
        c_obj = model.addConstrs((quicksum(
            self.c[i,j]*self.x[i,j,k] for i in self.N for j in self.N if i != j)
            <= P_max for k in self.K), name="c_obj")
        #model.write("x.lp")
        # Constraints
        # For detail on what the constraints signify please see the
        # corresponding Ipython Notebook
        c2_1 = self.model.addConstr((quicksum(
            x[s, j, k] for j in self.N for k in self.K for s in self.S if j not in self.S)
            == self.noOfRobots), name="c2_1")
        c2_2 = self.model.addConstr((quicksum(
            x[i, e, k] for i in self.N for k in self.K for e in self.E if i != e)
            == self.noOfRobots), name="c2_2")

        c3_1 = self.model.addConstrs(((quicksum(
            x[s, j, k] for s in self.S for j in self.N if j not in self.S))
            == 1 for k in self.K), name="c3_1")
        c3_2 = self.model.addConstrs(((quicksum(
            x[j, s, k] for s in self.S for j in self.N if j not in self.S))
            == 0 for k in self.K), name="c3_2")
        c3_3 = self.model.addConstrs(((quicksum(
            x[i, e, k] for e in self.E for i in self.N if i not in self.E))
            == 1 for k in self.K), name="c3_3")
        c3_4 = self.model.addConstrs(((quicksum(
            x[e, i, k] for e in self.E for i in self.N if i not in self.E))
            == 0 for k in self.K), name="c3_4")

        c4 = self.model.addConstrs(
            (quicksum(y[i, k] for k in self.K) <= 1 for i in self.T), name="c4")

        c5_1 = self.model.addConstrs(((quicksum(
            x[i, h, k] for i in self.N if i != h and i not in self.E)) ==
            (quicksum(x[h, j, k]
                      for j in self.N if j != h and j not in self.S))
            for h in self.N if h not in self.S+self.E for k in self.K), name="c5_1")
        c5_2 = self.model.addConstrs(((quicksum(
            x[i, h, k] for i in self.N if i != h)) ==
            y[h, k] for h in self.T for k in self.K), name="c5_2")
        c5_3 = self.model.addConstrs(((quicksum(
            x[h, j, k] for j in self.N if j != h)) ==
            y[h, k] for h in self.T for k in self.K), name="c5_3")

        c6 = self.model.addConstrs(((quicksum(
            (p[s, i, k] - p[i, s, k]) for i in self.N for s in self.S if i not in self.S)) ==
            (quicksum(x[i, j, k]
                      for i in self.T for j in self.N if i != j and i not in self.S))
            for k in self.K), name="c6")

        c7 = self.model.addConstrs(((quicksum(
            (p[j, i, k] - p[i, j, k]) for j in self.N if i != j)) ==
            (quicksum(x[i, j, k] for j in self.N if i != j))
            for i in self.T for k in self.K if i not in self.S), name="c7")

        c8 = self.model.addConstrs(((quicksum((
            p[j, i, k] - p[i, j, k]) for j in self.N if j != i))
            == 0 for i in self.D for k in self.K), name="c8")

        c9 = self.model.addConstrs((0 <= p[i, j, k] <= self.noOfTasks*x[i, j, k]
                                    for i in self.N for j in self.N for k in self.K if i != j), name="c9")

        M = 1e6
        c10 = self.model.addConstrs((r[j] - r[i] + self.f[i, j] <= M*(1-x[i, j, k])
                                     for i in self.T for j in self.T for k in self.K if i != j), name="c10")

        c11 = self.model.addConstrs((r[j] - r[i] + self.f[i, j] >= -M*(1-x[i, j, k])
                                     for i in self.T for j in self.T for k in self.K if i != j), name="c11")

        c12 = self.model.addConstrs((r[j] - self.L + self.f[i, j] <= M*(1-x[i, j, k])
                                     for i in self.D+self.S for j in self.T for k in self.K), name="c12")

        c13 = self.model.addConstrs((r[j] - self.L + self.f[i, j] >= -M*(1-x[i, j, k])
                                     for i in self.D+self.S for j in self.T for k in self.K), name="c13")

        c14 = self.model.addConstrs((r[i] - self.f[i, j] >= -M*(1-x[i, j, k])
                                     for i in self.T for j in self.D+self.E for k in self.K), name="c14")

        c15 = self.model.addConstrs(
            (0 <= r[i] <= self.L for i in self.T), name="c15")
        c16 = self.model.addConstrs(
            (self.f[i, j]*x[i, j, k] <=
             self.L for i in self.D for j in self.D+self.S+self.E for k in self.K if i != j), name="c16")

        c17 = self.model.addConstrs((
            quicksum(self.c[i, j]*x[i, j, k]*1 /
                     self.vel for i in self.N for j in self.N if i != j) <= self.T_max
            for k in self.K), name="c17")

    def solve(self):
        self.model.params.Heuristics = 0.0  # %age of time use a heuristic solution
        self.model.params.Cuts = 0  # Do not use cuts, except lazy constraints
        # model.params.MIPGapAbs = 0.0005
        # self.model.params.TimeLimit = 30
        self.model.optimize()

        return self.model 

    def write_lp_and_sol_to_disk(self):
        if not os.path.exists(self.instance_folder_path):
            os.makedirs(self.instance_folder_path)
        # Write both the LP file and the solution file
        self.model.write(self.file_path+'.lp')
        self.model.write(self.file_path+'.sol')


def main():
    min_robots = 5
    max_robots = 5

    min_depots = 3
    max_depots = 3

    min_tasks = 8
    max_tasks = 8

    fuel_range_start = 50
    # fuel_range_end = int(math.ceil(2*100*math.sqrt(2)/5)*5)
    fuel_range_end = 50
    fuel_range_step = 5

    Tmax_range_start = 150
    # Tmax_range_end = int(math.ceil(2*100*math.sqrt(2)/10)*10)
    Tmax_range_end = 150
    Tmax_range_step = 10

    robots_range = list(range(min_robots, max_robots+1))
    depots_range = list(range(min_depots, max_depots+1))
    tasks_range = list(range(min_tasks, max_tasks+1))
    fuel_range = list(range(fuel_range_start, fuel_range_end +
                            fuel_range_step, fuel_range_step))
    Tmax_range = list(range(Tmax_range_start, Tmax_range_end +
                            Tmax_range_step, Tmax_range_step,))

    no_of_instances = 5
    path_to_data_folder = os.getcwd()
    # instance_dictionary = {}

    for r in robots_range:
        for d in depots_range:
            for t in tasks_range:
                for f in fuel_range:
                    for tmax in Tmax_range:
                        instance_folder_path_suffix = \
                            '/data' + \
                            '/R' + str(r) + \
                            '/D' + str(d) + \
                            '/T' + str(t) + \
                            '/F' + str(f) + \
                            '/Tmax' + str(tmax)
                        instance_folder_path = os.path.normpath(
                            path_to_data_folder + instance_folder_path_suffix)
                        instance_filename_prefix = '/R' + str(r) + \
                            'D' + str(d) + \
                            'T' + str(t) + \
                            'F' + str(f) + \
                            'Tmax' + str(tmax)
                        for it in range(no_of_instances):
                            curr_instance_filename = instance_filename_prefix + \
                                'Iter' + str(it) + '.json'
                            file_path = os.path.normpath(
                                instance_folder_path+curr_instance_filename)
                            instance = InstanceReader(file_path)
                            instance_data = instance.readData()
                            solver = Solver(instance_data)
                            model = solver.solve()
                            solver.write_lp_and_sol_to_disk()


if __name__ == "__main__":
    main()
