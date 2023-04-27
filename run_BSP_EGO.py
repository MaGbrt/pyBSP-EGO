#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:55:57 2023

@author: maxime

run with:
    mpiexec -n 4 python3 run_BSP_EGO.py
"""
from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
n_proc = comm.Get_size()
print('From ', my_rank, ' : Running main with ', n_proc, 'proc')
import numpy as np
import parameters

import matplotlib.pyplot as plt
import numpy as np
from Problems.Rosenbrock import Rosenbrock
from Problems.Alpine2 import Alpine2
from Problems.Ackley import Ackley
from Problems.Schwefel import Schwefel
from Problems.CEC2014 import CEC2014
from Problems.Hybrid1 import Hybrid1
from Problems.Composition1 import Composition1

from Full_loops.parallel_l1_BSP_EGO_cycle import par_l1_BSP_EGO_run

from DataSets.DataSet import DataBase
from random import random
 
# Budget parameters
dim = 6;
batch_size = n_proc;
budget = 16;
print('The budget for this run is:', budget, ' cycles.')
t_max = parameters.max_time; # seconds
n_init = 96
n_cycle = budget
n_leaves = 2*n_proc
tree_depth = int(np.log(n_leaves)/np.log(2))
n_init_leaves = pow(2, tree_depth)
n_learn = min(n_init, 128)
   
# Define the problem
f = Ackley(dim)
f.set_default_bounds()


# Initialize Data
folder = 'Results/'
ext = '.txt'

if (my_rank == 0):
    DB = DataBase(f, n_init)
    r = random()*1000
    DB.par_create(comm = comm, seed = r)
        
    target, time = par_l1_BSP_EGO_run(DB, n_cycle, t_max, batch_size, tree_depth, n_learn, 0, comm)
    del DB
    # plt.plot(target[0], np.arange(budget+1))
    # plt.title('Evolution of the current outcome')
else:
    DB_worker = DataBase(f, n_init) # in order to access function evaluation
    DB_worker.par_create(comm = comm)

    par_l1_BSP_EGO_run(DB_worker, n_cycle, None, batch_size, None, n_learn, None, comm)
    del DB_worker

print('END', my_rank)
