from gurobipy import *
import random as rnd
import math
import numpy as np
import os
from TO_InstanceReader import InstanceReader
from TO_SysPathGenerator import SysPathGenerator


class F7Solver:

    def __init__(self, instance):
        '''
        param instance: A single instance that can be solved by the MinMax Solver
        '''
        self.iteration = instance.iteration
        self.noOfRobots = instance.noOfRobots
        self.noOfTasks = instance.noOfTasks
        self.noOfDepots = instance.noOfDepots
        self.L = instance.L
        self.delta = instance.delta
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
        #Adding stuff here
        # self.sum_of_all_task_distances_from_origin = instance.sum_of_all_task_distances_from_origin
        # self.mean_task_distance = instance.mean_task_distance
        self.sum_of_all_task_distances_from_origin = 0.0
        for i in range (self.noOfTasks):
            x = self.T_loc["T" + str(i)][0]
            y = self.T_loc["T" + str(i)][1]
            self.sum_of_all_task_distances_from_origin += math.sqrt (x**2 + y**2)
        # self.mean_task_distance = self.sum_of_all_task_distances_from_origin/self.noOfTasks
        self.initial_noOfRobots = self.noOfRobots
        self.noOfRobots = max(1,1+int(self.sum_of_all_task_distances_from_origin / self.delta ))
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
        self.filePaths = SysPathGenerator(self.noOfRobots, self.noOfDepots, self.noOfTasks, 
                                            self.delta, self.T_max)
        self.instance_folder_path = self.filePaths.instance_sol_folder_path
        self.instance_filename_prefix = '/F7' + self.filePaths.instance_sol_filename_prefix

        self.curr_instance_filename = self.instance_filename_prefix + \
            'Iter' + str(self.iteration)
        self.file_path = os.path.normpath(
            self.instance_folder_path + self.curr_instance_filename)

        # Initialize the model
        self.init_model()

    def init_model(self):
        # Initialize the model
        self.model = Model(
            'F7TOBasic-'+self.curr_instance_filename[1:]+'-Seed:' + str(self.thisSeed))
        # Decision variables and their bounds
        x = self.model.addVars(self.arcs, lb = 0, ub = self.arc_ub, name="x", vtype=GRB.INTEGER)
        y = self.model.addVars(self.k_y, name="y", vtype=GRB.BINARY)
        r = self.model.addVars(self.N, lb=0, ub=self.L, vtype=GRB.CONTINUOUS, name="r")
        q = self.model.addVars(self.arcs, vtype=GRB.CONTINUOUS, name="q")
        z = self.model.addVar(name="z", vtype=GRB.CONTINUOUS)
        
        # Objective function
        gamma = 1e-4
        objExpr1 = quicksum(y[i,k] for i in self.T for k in self.K)
        objExpr2 = gamma*z
        objFun = objExpr1 - objExpr2
        self.model.setObjective(objFun, GRB.MAXIMIZE)

        # Constraints
        # For detail on what the constraints signify please see the
        # corresponding section in the report
        c1 = self.model.addConstrs(((quicksum(self.c[i,j]*x[i,j,k] for i in self.N for j in self.N if i!=j)) <= z for k in self.K ), name="c1")

        #c2_1 = self.model.addConstr((quicksum(x[s,j,k] for j in self.N for k in self.K for s in self.S if j not in self.S) == self.noOfRobots), name="c2_1")
        #c2_2 = self.model.addConstr((quicksum(x[i,e,k] for i in self.N for k in self.K for e in self.E if i!=e) == self.noOfRobots), name="c2_2")
        
        c3_1 = self.model.addConstrs(((quicksum(x[s,j,k] for s in self.S for j in self.N if j not in self.S)) == 1 for k in self.K), name="c3_1")
        c4_1 = self.model.addConstrs(((quicksum(x[j,s,k] for s in self.S for j in self.N if j not in self.S)) == 0 for k in self.K), name="c3_2")
        c3_2 = self.model.addConstrs(((quicksum(x[i,e,k] for e in self.E for i in self.N if i not in self.E)) == 1 for k in self.K), name="c4_1")
        c4_2 = self.model.addConstrs(((quicksum(x[e,i,k] for e in self.E for i in self.N if i not in self.E)) == 0 for k in self.K), name="c4_2")

        c5_1 = self.model.addConstrs(((quicksum(x[i,h,k] for i in self.N if i!=h and i not in self.E)) == (quicksum(x[h,j,k] for j in self.N if j!=h and j not in self.S)) 
                                            for h in self.N for k in self.K if h not in self.S and h not in self.E), name="c5_1")
        c5_2 = self.model.addConstrs(((quicksum(x[i,h,k] for i in self.N if i!=h)) == y[h,k] for h in self.T for k in self.K), name="c5_2")
        c5_3 = self.model.addConstrs(((quicksum(x[h,j,k] for j in self.N if j!=h)) == y[h,k] for h in self.T for k in self.K), name="c5_3")
        c5_4 = self.model.addConstrs((quicksum(y[i,k] for k in self.K) <= 1 for i in self.T), name="c5_4")

        c6 = self.model.addConstrs((quicksum(self.c[i,j]*x[i,j,k]*1/self.vel for i in self.N for j in self.N if i!=j) <= self.T_max 
                                                        for k in self.K), name="c6")
        
        # Time based flow constraints
        c7 = self.model.addConstrs((quicksum(q[i,j,k] for j in self.N if i!=j and j not in self.S) - 
                         quicksum(q[j,i,k] for j in self.N if i!=j and j not in self.E) == 
                                quicksum(self.c[i,j]*x[i,j,k] for j in self.N if i!=j) 
                                            for k in self.K  for i in self.N if i not in self.E), name='c7' )
        #c8 = self.model.addConstrs((0 <= q[i,j,k] <= self.T_max*x[i,j,k] for i in self.N for j in self.N for k in self.K if i!=j and i not in self.E and j not in self.S), name='c8' )
        c8_1 = self.model.addConstrs((0 <= q[i,j,k] for i in self.N for j in self.N for k in self.K if i!=j and i not in self.E and j not in self.S), name='c8_1' )
        c8_2 = self.model.addConstrs((q[i,j,k] <= self.T_max*x[i,j,k] for i in self.N for j in self.N for k in self.K if i!=j and i not in self.E and j not in self.S), name='c8_2' )
        c9 = self.model.addConstrs((q[s,i,k] == self.f[s,i]*x[s,i,k] for i in self.T+self.D+self.E for s in self.S for k in self.K), name='c9')

        
        # Node based fuel constraints
        M = 1e6
        c10 = self.model.addConstrs((r[j] - r[i] + self.f[i,j] <= M*(1-x[i,j,k]) for i in self.T for j in self.T for k in self.K if i!=j), name="c10")
        c11 = self.model.addConstrs((r[j] - r[i] + self.f[i,j] >= -M*(1-x[i,j,k]) for i in self.T for j in self.T for k in self.K if i!=j), name="c11")
        c12 = self.model.addConstrs((r[j] - self.L + self.f[i,j] <= M*(1-x[i,j,k]) for i in self.D+self.S for j in self.T+self.D+self.E for k in self.K if i!=j), name="c12")
        c13 = self.model.addConstrs((r[j] - self.L + self.f[i,j] >= -M*(1-x[i,j,k]) for i in self.D+self.S for j in self.T+self.D+self.E for k in self.K if i!=j), name="c13")
        c14 = self.model.addConstrs((r[i] - self.f[i,j] >= -M*(1-x[i,j,k]) for i in self.S+self.T for j in self.D+self.E for k in self.K ), name="c14")
        #c15 = self.model.addConstrs((0 <= r[i] <= self.L for i in self.T+self.E), name="c16")
        c15_1 = self.model.addConstrs((0 <= r[i] for i in self.T+self.E), name="c15_1")
        c15_2 = self.model.addConstrs((r[i] <= self.L for i in self.T+self.E), name="c15_2")
        c16 = self.model.addConstrs((self.f[i,j]*x[i,j,k] <= self.L for i in self.S+self.D for j in self.D for k in self.K if i!=j), name="c16")


    def solve(self):
        #self.model.params.Heuristics = 0.0  # %age of time use a heuristic solution
        #self.model.params.Cuts = 0  # Do not use cuts, except lazy constraints
        # model.params.MIPGapAbs = 0.0005
        self.model.Params.MIPGap = 1e-3
        # self.model.params.TimeLimit = 30
        self.model.optimize()

        return self.model 

    def write_lp_and_sol_to_disk(self):
        if not os.path.exists(self.instance_folder_path):
            os.makedirs(self.instance_folder_path)
        # Save runtime, because after writing the lp file, 
        # runtime is lost. No idea why
        run_time = self.model.Runtime
        print("************ Run Time******", run_time)
        print("************ Sum of distances ", self.sum_of_all_task_distances_from_origin)
        # print("************ Mean Dist", self.mean_task_distance)
        print("************ No of Tasks", self.noOfTasks)
        print("************ Initial No of robots", self.initial_noOfRobots)
        print("************ No of robots ", self.noOfRobots)
        # Write both the LP file and the solution file
        self.model.write(self.file_path+'.lp')
        self.model.write(self.file_path+'.sol')
        # Add runtime in the sol file as well.
        with open(self.file_path+'.sol', "a") as myfile:
            myfile.write('runtime:{}'.format(run_time))


def main():
    min_robots = 5

    max_robots = min_robots

    min_depots = 1
    max_depots = 1

    min_tasks = 20

    max_tasks = min_tasks

    delta_range_start = 300
    delta_range_step = 100
    # delta_range_end = int(math.ceil(2*100*math.sqrt(2) /
    #                               delta_range_step)*delta_range_step)  # ~282
    delta_range_end = 300

    Tmax_range_start = 600
    Tmax_range_step = 100
    # Tmax_range_end = int(math.ceil(2*100*math.sqrt(2) /
    #                               Tmax_range_step)*Tmax_range_step)  # ~282
    Tmax_range_end = 600

    robots_range = list(range(min_robots, max_robots+1))
    depots_range = list(range(min_depots, max_depots+1))
    tasks_range = list(range(min_tasks, max_tasks+1))
    delta_range = list(range(delta_range_start, delta_range_end +
                            delta_range_step, delta_range_step))
    Tmax_range = list(range(Tmax_range_start, Tmax_range_end +
                            Tmax_range_step, Tmax_range_step,))

    iter_no_list = [0] #1.
    path_to_data_folder = os.getcwd()
    # instance_dictionary = {}

    for r in robots_range:
        for d in depots_range:
            for t in tasks_range:
                for f in delta_range:
                    for tmax in Tmax_range:
                        filePaths = SysPathGenerator(r, d, t, f, tmax)
                        instance_folder_path = filePaths.instance_data_folder_path
                        instance_filename_prefix = filePaths.instance_data_filename_prefix

                        for it in iter_no_list:
                            curr_instance_filename = instance_filename_prefix + \
                                'Iter' + str(it) + '.json'
                            file_path = os.path.normpath(
                                instance_folder_path+curr_instance_filename)
                            instance = InstanceReader(curr_instance_filename)
                            instance_data = instance.readData()
                            solver = F7Solver(instance_data)
                            model = solver.solve()
                            solver.write_lp_and_sol_to_disk()

if __name__ == "__main__":
    main()
