#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:22:36 2022

@author: maxime
"""

import numpy as np
import torch
import gpytorch
import parameters
from Models.GPyTorch_models import GP_model, SKIP_GP_model
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.settings import cholesky_jitter

from BSP_tree.tree import Tree
from BSP_tree.split_functions import default_split
from time import time

dtype = torch.double

def par_l1_BSP_EGO_run(DB, n_cycle, t_max, batch_size, tree_depth, n_learn, id_run, comm):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    tol = 0.01

    dim = DB._dim
    if my_rank == 0:
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = torch.min(DB._y).numpy()

        bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])
        T = Tree(DB, bounds)
        T.build(depth = tree_depth, split_function = default_split)
        
        time_per_cycle = np.zeros((n_cycle, 5))
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        iter = 0
        t_0 = time()
        t_current = 0.
        while (iter < n_cycle and t_current < t_max):
            keep_going = np.ones(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)
            t_start = time()
            comm.Bcast(DB._X.numpy(), root = 0)
            comm.Bcast(DB._y.numpy(), root = 0)

            subdomains = T.get_leaves()
            n_leaves = subdomains.get_size()
            T.check_volume()

            n_tasks = np.zeros(n_proc, dtype = 'i')
            for c in range(subdomains.get_size()):
                send_to = c%n_proc
                n_tasks[send_to] += 1
            comm.Bcast(n_tasks, root = 0)
            
            my_tasks = n_tasks[my_rank]
            b_list = []
            for c in range(n_leaves):
                send_to = c%n_proc
                if send_to == 0:
                    bounds = subdomains._list[c]._domain
                    b_list.append(bounds)
                else:
                    bounds = subdomains._list[c]._domain
                    comm.send(bounds, dest = send_to, tag = c)

            t_model = time ()
            candidates = np.zeros((my_tasks, DB._dim + 1))
            t_mod = 0
            t_ap_sum = 0
            for t in range(my_tasks):
                bounds = b_list[t]
                center = (bounds[0]+bounds[1])*0.5
                DB_temp = DB.copy_from_center(n_learn, center)
                scaled_y = DB_temp.min_max_y_scaling()

                if(torch.isnan(scaled_y).any() == True):
                    print(my_rank, ' iter ', iter, ' task ', t, ' DB temp X: \n', DB_temp._X)
                    print(my_rank, ' iter ', iter, ' task ', t, ' DB temp y: \n', DB_temp._y)
                    print(my_rank, ' iter ', iter, ' task ', t,' scaled train Y: \n', scaled_y)
                
                model = GP_model(DB_temp._X, scaled_y)
                with gpytorch.settings.max_cholesky_size(parameters.max_cholesky_size):
                    t_mod0 = time()
                    model.custom_fit(parameters.large_fit)  
                    t_mod += time() - t_mod0
                    
                    crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
                    t_ap0 = time()
                    try :
                        candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=parameters.af_nrestarts, raw_samples=parameters.af_nsamples, options=parameters.af_options)
                    except :
                        print('\n Failed to optimize the acquisition function. Why ?')
                        #DB_temp.try_distance(tol)
                        print('try again with new model an increase of the jitter')
                        model = GP_model(DB_temp._X, scaled_y)
                        with gpytorch.settings.max_cholesky_size(parameters.max_cholesky_size):
                            model.custom_fit(parameters.large_fit)  

                        with cholesky_jitter(parameters.chol_jitter):
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=parameters.af_nrestarts, raw_samples=parameters.af_nsamples, options=parameters.af_options)
                    t_ap_sum += time() - t_ap0
                                
                candidates[t, 0:dim] = candidate.numpy()
                candidates[t, dim] = acq_value.numpy()
                
                del model
                del DB_temp
            time_per_cycle[iter, 0] = t_mod
            time_per_cycle[iter, 1] = t_ap_sum

            candidates_list = []
            acq_value_list = []
            cpt = 0
            for c in range(n_leaves):
                get_from = c%n_proc
                if (get_from == 0):
                    cand = candidates[cpt,:]
                    cpt += 1
                else :
                    cand = comm.recv(source = get_from, tag = c)

                candidates_list.append(cand[0:dim])
                acq_value_list.append(-cand[dim])
                subdomains._list[c]._crit = acq_value_list[c]
            
            try :
                sort_X = [el for _, el in sorted(zip(acq_value_list, candidates_list), reverse = True)]
            except :
                print('Failed to sort list')
                sort_X = candidates_list
            selected_candidates = DB.select_clean(sort_X, batch_size, n_leaves, tol)
            t_ap = time()
            time_per_cycle[iter, 2] = t_ap - t_start

            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(selected_candidates)):
                send_to = c%n_proc
                n_cand[send_to] += 1
            comm.Bcast(n_cand, root = 0)
            for c in range(len(selected_candidates)):
                send_cand = selected_candidates[c]
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = 1000 + c)
            for c in range(int(n_cand[0])):
                y_new = DB._obj.evaluate(selected_candidates[n_proc*c])

                DB.add(torch.tensor(selected_candidates[n_proc*c]).reshape(1,dim), torch.tensor(y_new))

            for c in range(len(selected_candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = 1000 + c)
                    DB.add(torch.tensor(selected_candidates[c]).reshape(1,dim), torch.tensor(recv_eval))
        
            target[0, iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 3] = t_end - t_ap
            time_per_cycle[iter, 4] = t_end - t_start
            arg_min = torch.argmin(DB._y)

            T.update(subdomains)
            
            print("Alg. lBSP-EGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ',  DB._y[arg_min])
            DB.print_min()
            if(parameters.sim_cost == -1):
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[iter, 2] + parameters.sim_cost
            print('t_current: ', t_current)
            print('real time is ', time() - t_0)
            iter = iter + 1
        else :
            print('Budget is out, time is %.6f' %t_current)
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)

        DB.print_min()
        del T
        return target, time_per_cycle
                    
    else:
        init_size = DB._size
        for iter in range(n_cycle+1):
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)
            
            if(keep_going.sum() == 1):
                tmp_X = np.ndarray((init_size + iter * batch_size, dim))
                tmp_y = np.ndarray((init_size + iter * batch_size, 1))
                comm.Bcast(tmp_X, root = 0)
                comm.Bcast(tmp_y, root = 0)
    
                DB.set_Xy(tmp_X, tmp_y)
                
                n_tasks = np.zeros(n_proc, dtype = 'i')
                comm.Bcast(n_tasks, root = 0)                
                my_tasks = n_tasks[my_rank] 
                b_list = []
                bounds = np.zeros((2, DB._dim))
                for t in range(my_tasks):
                    bounds = comm.recv(source = 0, tag = my_rank + t * n_proc)
                    b_list.append(bounds)
                candidates = np.zeros((my_tasks, DB._dim + 1))
                for t in range(my_tasks):
                    b_temp = b_list[t]
                    center = (b_temp[0]+b_temp[1])*0.5
                    DB_temp = DB.copy_from_center(n_learn, center)
                    scaled_y = DB_temp.min_max_y_scaling()
                    if(torch.isnan(scaled_y).any() == True):
                        print(my_rank, ' iter ', iter, ' task ', t, ' DB temp X: \n', DB_temp._X)
                        print(my_rank, ' iter ', iter, ' task ', t, ' DB temp y: \n', DB_temp._y)
                        print(my_rank, ' iter ', iter, ' task ', t, ' scaled train Y: \n', scaled_y)
    
                    model = GP_model(DB_temp._X, scaled_y)
                    with gpytorch.settings.max_cholesky_size(parameters.max_cholesky_size):
                        model.custom_fit(parameters.large_fit)  
                        crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
                        try :
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=parameters.af_nrestarts, raw_samples=parameters.af_nsamples, options=parameters.af_options)
                        except :
                            print('\n Failed to optimize the acquisition function. Why ?')
                            print('try again with new model an increase of the jitter')
                            model = GP_model(DB_temp._X, scaled_y)
                            model.custom_fit(parameters.large_fit)  

                            with cholesky_jitter(parameters.chol_jitter):
                                candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=parameters.af_nrestarts, raw_samples=parameters.af_nsamples, options=parameters.af_options)

                    candidates[t, 0:dim] = candidate.numpy()
                    candidates[t, dim] = acq_value.numpy()
                    
                    del model
                    del DB_temp
                
                for t in range(my_tasks):
                    comm.send(candidates[t,], dest = 0, tag = my_rank + t * n_proc)
    
                n_cand = np.zeros(n_proc, dtype = 'i')
                comm.Bcast(n_cand, root = 0)
                cand = []
                for c in range(n_cand[my_rank]):
                    cand.append(comm.recv(source = 0, tag = 1000 + my_rank + c * n_proc))
    
                y_new = []
                for c in range(n_cand[my_rank]):
                    y_new.append(DB._obj.evaluate(cand[c]))
    
                for c in range(n_cand[my_rank]):
                    comm.send(y_new[c], dest = 0, tag = 1000 + my_rank + c * n_proc)
            else : 
                break
                
        return None        

