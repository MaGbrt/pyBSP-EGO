#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:30:23 2022

@author: maxime
"""
sim_cost = -1 #5 #float(-1.) # seconds (-1 if real cost)
max_time = 20*60 #sim_cost * 80 # seconds, maximum time if budget in seconds
max_iter = 100000000 # iterations, maximum number of iterations if budget in simulations
max_cholesky_size = float("inf")  # Always use Cholesky
chol_jitter = 1e-3

large_fit = 500
small_fit = 50
medium_fit = 200

af_nrestarts = 10
af_nsamples = 512
af_options = {}
af_options['maxfun']=500
af_options['iprint']=-1 # neg means no output
af_options['method']='L-BFGS-B'