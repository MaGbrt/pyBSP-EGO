import numpy as np
from Problems.Problem import Problem
from numpy import ma
class Hybrid1(Problem):
    """Class for the single-objective Hybrid1 problem.

    :param dim: number of decision variable
    :type dim: positive int, >1
    """

    #-------------__init__-------------#    
    def __init__(self, dim):
        assert dim>=1
        self._dim = dim
        self._bounds = np.zeros((2,dim))
        self._name = 'Hybrid1'
        print('Initialise Hybrid1 function with ' + str(self._dim) + ' dimensions')
        
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
        
#        candidates = candidates * (self.get_bounds()[1] - self.get_bounds()[0]) + self.get_bounds()[0]
        candidates = self.unmap(candidates)
        assert self.is_feasible(candidates)
        
        if candidates.ndim==1:
            candidates = np.array([candidates])

        M_p = np.load('M_p.npy')# matrice de permutation
        k = 0
        for cand in candidates:
            candidates[k] = M_p @ cand + 3.45
            k+=1
            
        alpine_mask = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1], bool)
        schwefel_mask = ~alpine_mask    

        a = np.arange(len(alpine_mask))
        rm_alp = a[~alpine_mask]
        rm_sch = a[~schwefel_mask]
        
        c_alp = np.delete(candidates, rm_alp, 1)
        c_alp = (c_alp + 100)/20 # Scale into [0, 10]^D, default for Alpine2, instead of [-100, 100] for CEC2015

        c_sch = np.delete(candidates, rm_sch, 1)

        obj_vals1 = np.prod(np.sqrt(c_alp)*np.sin(c_alp), axis=1)        
        obj_vals2 = 418.9828872724338*len(rm_alp)-np.einsum('ij,ij->i', c_sch, np.sin(np.sqrt(np.abs(c_sch))))

        
        obj_vals = obj_vals1 + obj_vals2
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
