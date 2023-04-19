#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 17:42:28 2022

@author: gobertm
"""
from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
n_proc = comm.Get_size()
print('From ', my_rank, ' : Running main with ', n_proc, 'proc')
import numpy as np
import sys
import parameters
from Problems.Rosenbrock import Rosenbrock
from Problems.Alpine2 import Alpine2
from Problems.Ackley import Ackley
from Problems.Schwefel import Schwefel
from Problems.CEC2014 import CEC2014
from Problems.Hybrid1 import Hybrid1
from Problems.Composition1 import Composition1

from Full_loops.parallel_EGO_cycle import par_EGO_run, par_MC_qEGO_run, par_MCbased_qEGO_run
from Full_loops.parallel_g1_BSP_EGO_cycle import par_g1_BSP_EGO_run, par_g1_BSP_qEGO_run, par_g1_BSP2_EGO_run
from Full_loops.parallel_l1_BSP_EGO_cycle import par_l1_BSP_EGO_run, par_lg1_BSP_EGO_run, par_lg2_BSP_EGO_run, par_l2_BSP_EGO_run
from Full_loops.parallel_TuRBO1_cycle import par_Turbo1_run

from Full_loops.parallel_random_sampling import par_random_run

from DataSets.DataSet import DataBase
from random import random
 
# Budget parameters
DoE_num = int(sys.argv[1]);
dim = int(sys.argv[3]);
batch_size = n_proc;
budget = int(parameters.max_time / parameters.sim_cost);
print('The budget for this run is:', budget, ' cycles.')
t_max = parameters.max_time; # seconds
n_init = 96#(int) (min((0.2*budget)*batch_size, 256));
n_cycle = budget#(int) (0.8*budget*4);
n_leaves = 2*n_proc
tree_depth = int(np.log(n_leaves)/np.log(2))
n_init_leaves = pow(2, tree_depth)
#max_leaves = 4 * n_init_leaves
n_learn = min(n_init, 128)
#n_learn = n_init
   
# Define the problem
dict_p = {}
dict_p['Ackley'] = Ackley(dim)
dict_p['Alpine2'] = Alpine2(dim)
dict_p['Rosenbrock'] = Rosenbrock(dim)
dict_p['Schwefel'] = Schwefel(dim)
dict_p['Hybrid1'] = Hybrid1(dim)
dict_p['Composition1'] = Composition1(dim)

if (sys.argv[2] == 'CEC2014'):
    f_id = int(sys.argv[5])
    dict_p['CEC2014'] = CEC2014(dim, f_id)

if (sys.argv[2] == 'UPHES'):
    assert dim == 12, f"Dimension for UPHES problem should be 12, got: {dim}"
    from Problems.UPHES import UPHES
    dict_p['UPHES'] = UPHES(dim, my_rank)

f = dict_p[sys.argv[2]]
f.set_default_bounds()

# dict_alg = {}
# dict_alg['random'] = par_random_run
# dict_alg['turbo'] = par_Turbo1_run
# dict_alg['KBqEGO'] = par_EGO_run
# dict_alg['MCqEGO'] = par_MC_qEGO_run
# dict_alg['MCbasedqEGO'] = par_MCbased_qEGO_run
# dict_alg['gBSPEGO'] = par_g1_BSP_EGO_run
# dict_alg['lBSPEGO'] = par_l1_BSP_EGO_run

# run = dict_alg[sys.argv[3]]

run_random = False
run_turbo_ei = False
run_qEGO = False
run_MC_qEGO = False 
run_MCbased_qEGO = False
run_gBSP_EGO = False
run_gBSP_qEGO = False
run_gBSP2_EGO = False
run_lBSP_EGO = False
run_lg1BSP_EGO = False
run_l2BSP_EGO = False

if (sys.argv[4] == "random"):
    run_random = True
if (sys.argv[4] == "turbo"):
    run_turbo_ei = True
if (sys.argv[4] == "KBqEGO"):
    run_qEGO = True
if (sys.argv[4] == "MCqEGO"):
    run_MC_qEGO = True
if (sys.argv[4] == "MCbasedqEGO"):
    run_MCbased_qEGO = True
if (sys.argv[4] == "gBSPEGO"):
    run_gBSP_EGO = True
if (sys.argv[4] == "gBSPqEGO"):
    run_gBSP_qEGO = True
if (sys.argv[4] == "gBSP2EGO"):
    run_gBSP2_EGO = True
if (sys.argv[4] == "lBSPEGO"):
    run_lBSP_EGO = True
if (sys.argv[4] == "lg1BSPEGO"):
    run_lg1BSP_EGO = True
if (sys.argv[4] == "l2BSPEGO"):
    run_l2BSP_EGO = True


# Initialize Data
#rep_vec = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
rep_vec = np.array([DoE_num])
n_rep = len(rep_vec)

folder = 'Results/'
ext = '.txt'
id_name = '_' + f._name + '_D' + str(dim) + '_batch' + str(batch_size) + '_budget' + str(budget) + '_t_cost' + str(parameters.sim_cost)

if (my_rank == 0):
    target_EGO = np.zeros((n_rep, n_cycle+1))
    target_MC_qEGO = np.zeros((n_rep, n_cycle+1))
    target_MCbased_qEGO = np.zeros((n_rep, n_cycle+1))
    target_Turbo_ei = np.zeros((n_rep, n_cycle+1))
    target_g1_BSP_EGO = np.zeros((n_rep, n_cycle+1))
    target_g1_BSP_qEGO = np.zeros((n_rep, n_cycle+1))
    target_g1_BSP2_EGO = np.zeros((n_rep, n_cycle+1))
    target_l1_BSP_EGO = np.zeros((n_rep, n_cycle+1))
    target_lg1_BSP_EGO = np.zeros((n_rep, n_cycle+1))
    target_l2_BSP_EGO = np.zeros((n_rep, n_cycle+1))
    target_random = np.zeros((n_rep, n_cycle+1))

#    for i_rep in range(n_rep):
    for i_rep in range(n_rep):
        k_rep = rep_vec[i_rep]
        # Input data scaled in [0, 1]^d
        DB = DataBase(f, n_init)
        r = random()*1000
        input_file = 'Initial_DoE' + id_name + '_run' + str(k_rep) + ext #None

        if (input_file == None):
            par_create = np.ones(1, dtype = 'i')
            comm.Bcast(par_create, root = 0)
            DB.par_create(comm = comm, seed = r)
            #            DB.create(seed=r)
#            DB.create_lhs(seed=r)
            full_name = 'Initial_DoE' + id_name + '_run' + str(k_rep) + ext
            DB.save_txt('DoE/' + full_name)
        else :
            par_create = np.zeros(1, dtype = 'i')
            comm.Bcast(par_create, root = 0)
            DB.read_txt('DoE/' + input_file)
        
        print('\n Synchronize before running Random')
        comm.Barrier()
        if run_random:
            DB_random = DB.copy()
            target_random[i_rep, :], time_random = par_random_run(DB_random, n_cycle, t_max, batch_size, DoE_num, comm)
            full_name = 'Random' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_random.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_random, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_random
        
        print('\n Synchronize before running Turbo')
        comm.Barrier()
        if run_turbo_ei:
            DB_Turbo_ei = DB.copy()
            target_Turbo_ei[i_rep, :], time_turbo_ei = par_Turbo1_run(DB_Turbo_ei, n_cycle, t_max, batch_size, "ei", DoE_num, comm)
            full_name = 'TuRBO' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_Turbo_ei.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_turbo_ei, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_Turbo_ei

        print('\n Synchronize before running qEGO from Ginsbourger et al.')
        comm.Barrier()
        if run_qEGO:
            DB_EGO = DB.copy()
            target_EGO[i_rep, :], time_EGO = par_EGO_run(DB_EGO, n_cycle, t_max, batch_size, DoE_num, 'ei', comm)
            full_name = 'KB_qEGO_ei' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_EGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_EGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_EGO
            
        print('\n Synchronize before running multi criteria qEGO')   
        comm.Barrier()
        if run_MC_qEGO:
            DB_MC_qEGO = DB.copy()
            target_MC_qEGO[i_rep, :], time_MC_qEGO = par_MC_qEGO_run(DB_MC_qEGO, n_cycle, t_max, batch_size, DoE_num, comm)
            full_name = 'MC_qEGO' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_MC_qEGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_MC_qEGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_MC_qEGO
            
        print('\n Synchronize before running Monte Carlo based qEGO')
        comm.Barrier()
        if run_MCbased_qEGO:
            DB_MCbased_qEGO = DB.copy()
            target_MCbased_qEGO[i_rep, :], time_MCbased_qEGO = par_MCbased_qEGO_run(DB_MCbased_qEGO, n_cycle, t_max, batch_size, DoE_num, comm)
            full_name = 'MCbased_qEGO' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_MCbased_qEGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_MCbased_qEGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_MCbased_qEGO
            
        print('\n Synchronize before running BSP-EGO with global model')
        comm.Barrier()
        if run_gBSP_EGO:
            DB_g1_BSP_EGO = DB.copy()
            target_g1_BSP_EGO[i_rep, :], time_g1_BSP_EGO = par_g1_BSP_EGO_run(DB_g1_BSP_EGO, n_cycle, t_max, batch_size, tree_depth, DoE_num, comm)
            full_name = 'Global_BSP_EGO' + id_name + '_t_max' + str(t_max) + '_depth' + str(tree_depth) + '_run' + str(k_rep) + ext
            DB_g1_BSP_EGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_g1_BSP_EGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_g1_BSP_EGO

        print('\n Synchronize before running BSP-qEGO with global model')
        comm.Barrier()
        if run_gBSP_qEGO:
            DB_g1_BSP_qEGO = DB.copy()
            target_g1_BSP_qEGO[i_rep, :], time_g1_BSP_qEGO = par_g1_BSP_qEGO_run(DB_g1_BSP_qEGO, n_cycle, t_max, batch_size, tree_depth, DoE_num, comm)
            full_name = 'Global_BSP_qEGO' + id_name + '_t_max' + str(t_max) + '_depth' + str(tree_depth) + '_run' + str(k_rep) + ext
            DB_g1_BSP_qEGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_g1_BSP_qEGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_g1_BSP_qEGO

        print('\n Synchronize before running BSP2-EGO with global model')
        comm.Barrier()
        if run_gBSP2_EGO:
            DB_g1_BSP2_EGO = DB.copy()
            target_g1_BSP2_EGO[i_rep, :], time_g1_BSP2_EGO = par_g1_BSP2_EGO_run(DB_g1_BSP2_EGO, n_cycle, t_max, batch_size, (tree_depth + 2), DoE_num, comm)
            full_name = 'Global_BSP22_EGO' + id_name + '_t_max' + str(t_max) + '_depth' + str((tree_depth+2)) + '_run' + str(k_rep) + ext
            DB_g1_BSP2_EGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_g1_BSP2_EGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_g1_BSP2_EGO

        print('\n Synchronize before running BSP-EGO with local models')
        comm.Barrier()
        if run_lBSP_EGO:
            print('run lBSP-EGO - master')
            DB_l1_BSP_EGO = DB.copy()
            target_l1_BSP_EGO[i_rep, :], time_l1_BSP_EGO = par_l1_BSP_EGO_run(DB_l1_BSP_EGO, n_cycle, t_max, batch_size, tree_depth, n_learn, DoE_num, comm)
            full_name = 'Local_BSP_EGO' + id_name + '_t_max' + str(t_max) + '_depth' + str(tree_depth) + '_run' + str(k_rep) + ext
            DB_l1_BSP_EGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_l1_BSP_EGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_l1_BSP_EGO
    
        print('\n Synchronize before running lBSP-EGO with local models and one global model')
        comm.Barrier()
        if run_lg1BSP_EGO:
            print('run lBSP-EGO - master')
            DB_lg1_BSP_EGO = DB.copy()
            target_lg1_BSP_EGO[i_rep, :], time_lg1_BSP_EGO = par_lg1_BSP_EGO_run(DB_lg1_BSP_EGO, n_cycle, t_max, batch_size, tree_depth, n_learn, DoE_num, comm)
            full_name = 'LG1_BSP_EGO' + id_name + '_t_max' + str(t_max) + '_depth' + str(tree_depth) + '_run' + str(k_rep) + ext
            DB_lg1_BSP_EGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_lg1_BSP_EGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_lg1_BSP_EGO
            
        print('\n Synchronize before running lBSP-EGO with local models and one global model')
        comm.Barrier()
        if run_l2BSP_EGO:
            print('run l2BSP-EGO - master')
            DB_l2_BSP_EGO = DB.copy()
            target_l2_BSP_EGO[i_rep, :], time_l2_BSP_EGO = par_l2_BSP_EGO_run(DB_l2_BSP_EGO, n_cycle, t_max, batch_size, tree_depth, n_learn, DoE_num, comm)
            full_name = 'L2_BSP_EGO' + id_name + '_t_max' + str(t_max) + '_depth' + str(tree_depth) + '_run' + str(k_rep) + ext
            DB_l2_BSP_EGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_l2_BSP_EGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_l2_BSP_EGO

        del DB
else:
    for i_rep in range(n_rep):
        par_create = np.zeros(1, dtype = 'i')
        comm.Bcast(par_create, root = 0)
        if par_create[0] == 1:
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            DB_worker.par_create(comm = comm)

        print('\n Synchronize before running random search - workers')
        comm.Barrier()
        if run_random:
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_random_run(DB_worker, n_cycle, None, batch_size, None, comm)
            del DB_worker
            
        print('\n Synchronize before running turbo - workers')
        comm.Barrier()
        if run_turbo_ei:
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_Turbo1_run(DB_worker, n_cycle, None, batch_size, None, None, comm)
            del DB_worker

        print('\n Synchronize before running qEGO - workers')
        comm.Barrier()
        if run_qEGO:
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_EGO_run(DB_worker, n_cycle, None, batch_size, None, None, comm)
            del DB_worker
    
        print('\n Synchronize before running mic qEGO - workers')
        comm.Barrier()
        if run_MC_qEGO:
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_MC_qEGO_run(DB_worker, n_cycle, None, batch_size, None, comm)
            del DB_worker
        
        print('\n Synchronize before running MC based qEGO - workers')
        comm.Barrier()
        if run_MCbased_qEGO:
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_MCbased_qEGO_run(DB_worker, n_cycle, None, batch_size, None, comm)
            del DB_worker

        print('\n Synchronize before running BSP-EGO with global model - workers')
        comm.Barrier()
        if run_gBSP_EGO:
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_g1_BSP_EGO_run(DB_worker, n_cycle, None, batch_size, None, None, comm)
            del DB_worker

        print('\n Synchronize before running BSP-qEGO with global model - workers')
        comm.Barrier()
        if run_gBSP_qEGO:
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_g1_BSP_qEGO_run(DB_worker, n_cycle, None, batch_size, None, None, comm)
            del DB_worker

        print('\n Synchronize before running BSP2-EGO with global model - workers')
        comm.Barrier()
        if run_gBSP2_EGO:
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_g1_BSP2_EGO_run(DB_worker, n_cycle, None, batch_size, None, None, comm)
            del DB_worker

        print('\n Synchronize before running BSP-EGO with local models - workers')
        comm.Barrier()
        if run_lBSP_EGO:
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_l1_BSP_EGO_run(DB_worker, n_cycle, None, batch_size, None, n_learn, None, comm)
            del DB_worker
            
        print('\n Synchronize before running lg1BSP-EGO with local models and one global model - workers')
        comm.Barrier()
        if run_lg1BSP_EGO:
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_lg1_BSP_EGO_run(DB_worker, n_cycle, None, batch_size, None, n_learn, None, comm)
            del DB_worker

        print('\n Synchronize before running l2BSP-EGO with local models and one global model - workers')
        comm.Barrier()
        if run_l2BSP_EGO:
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_l2_BSP_EGO_run(DB_worker, n_cycle, None, batch_size, None, n_learn, None, comm)
            del DB_worker
print('END', my_rank)
