import numpy as np
from Problems.Problem import Problem
from numpy import ma
from Problems.CEC2014 import CEC2014

class Composition1(Problem):
    """Class for the single-objective Composition1 problem.

    :param dim: number of decision variable
    :type dim: positive int, >1
    """

    #-------------__init__-------------#    
    def __init__(self, dim):
        assert dim>=1
        self._dim = dim
        self._bounds = np.zeros((2,dim))
        self._name = 'Composition1'
        self._f2 = CEC2014(self._dim, 6)
        self._f2.set_default_bounds()

        print('Initialise Composition1 function with ' + str(self._dim) + ' dimensions')
        
    #-------------__del__-------------#
    def __del__(self):
        Problem.__del__(self)

    #-------------evaluate-------------#
    def evaluate(self, candidates):
        """Objective function.

        :param candidates: candidate decision vectors
        :type candidates: np.ndarray
        :return: objective values
        :rtype: np.ndarray
        """
#       CEC2014 id 6 : Weierstrass function        
        c_wei = candidates
        obj_vals2 = self._f2.evaluate(c_wei)
        candidates = self.unmap(candidates)
        assert self.is_feasible(candidates)
        
        if candidates.ndim==1:
            candidates = np.array([candidates])

        
        M_p = np.load('M2_p.npy')# matrice de permutation
        shift = 13.23
        k = 0
        for cand in candidates:
            candidates[k] = M_p @ cand + shift
            k +=1
        c_alp = (candidates + 100)/20 # Scale from [-100; 100]^D into [0, 10]^D, default for Alpine2

        obj_vals1 = np.prod(np.sqrt(c_alp)*np.sin(c_alp), axis=1) # Alpine
        obj_vals = obj_vals1 * 0.
        
        # print('evaluating f1: \n', c_alp, ' = ', obj_vals1)
        # print('evaluating f2: \n', c_wei, ' = ', obj_vals2)

        for k in range(len(candidates)):
            cand = candidates[k]
            ss1 = np.sqrt(np.dot(cand-shift, cand-shift)) + 1
            ss2 = np.sqrt(np.dot(cand, cand)) + 1
            w1 = np.exp(-ss1/(2*self._dim*100))/ss1
            w2 = np.exp(-ss2/(2*self._dim*225))/ss2
            v1 = w1 / (w1+w2)
            v2 = w2 / (w1+w2)
            obj_vals[k] = v1 * 0.7 * obj_vals1[k] + v2 * 8*obj_vals2[k]
        
        #print('evaluating: \n', candidates, ' = ', obj_vals)

        return obj_vals

    #-------------set_bounds-------------#
    def set_bounds(self, bounds):
        """Set box-constrained search space bounds.
        :type bounds: np.ndarray
        :rtype: np.ndarray
        """
        self._bounds = bounds

    #-------------set_default_bounds-------------#
    def set_default_bounds(self):
        """Set box-constrained search space bounds.
        """
        b=np.ones((2,self._dim))
        b[0,:]*=-100
        b[1,:]*=100
        self._bounds = b

    #-------------get_bounds-------------#
    def get_bounds(self):
        """Returns search space bounds.

        :returns: search space bounds
        :rtype: np.ndarray
        """
        return self._bounds

    #-------------is_feasible-------------#
    def is_feasible(self, candidates):
        """Check feasibility of candidates.

        :param candidates: candidate decision vectors
        :type candidates: np.ndarray
        :returns: boolean indicating whether candidates are feasible
        :rtype: bool
        """

        res=False
        if Problem.is_feasible(self, candidates)==True:
            lower_bounds=self.get_bounds()[0,:]
            upper_bounds=self.get_bounds()[1,:]
            res=(lower_bounds<=candidates).all() and (candidates<=upper_bounds).all()
        return res

    def unmap(self, candidates):
        return candidates * (self.get_bounds()[1] - self.get_bounds()[0]) + self.get_bounds()[0]

    def my_map(self, candidates):
        return (candidates - self.get_bounds()[0]) / (self.get_bounds()[1] - self.get_bounds()[0])
