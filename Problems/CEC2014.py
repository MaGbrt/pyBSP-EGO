import numpy as np
from Problems.Problem import Problem
from pygmo import cec2014
import pygmo

class CEC2014(Problem):
    """Class for the single-objective CEC2014 problem.

    :param dim: number of decision variable
    :type dim: positive int, >1
    """

    #-------------__init__-------------#    
    def __init__(self, dim, f_id):
        assert dim>=1
        self._dim = dim
        self._bounds = np.zeros((2,dim))
        self._name = 'CEC2014_' + str(f_id)
        self._f_id = f_id
        
        self._f = pygmo.problem(cec2014(prob_id = self._f_id, dim = self._dim))
        print('Initialise CEC2014 function ' + str(f_id) + ' with ' + str(self._dim) + ' dimensions')
        print("Function's name: ", self._f.get_name())

        
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
            
        obj_vals = np.zeros((candidates.shape[0],))
        for i,cand in enumerate(candidates):
            obj_vals[i] = self._f.fitness(cand)
        
      #  print('evaluating: \n', candidates, ' = ', obj_vals)
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
        self._bounds[0, :] = self._f.get_bounds()[0]
        self._bounds[1, :] = self._f.get_bounds()[1]

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
