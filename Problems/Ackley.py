import numpy as np
from Problems.Problem import Problem

class Ackley(Problem):
    """Class for the single-objective Ackley problem.

    :param dim: number of decision variable
    :type dim: positive int, >1
    """

    #-------------__init__-------------#    
    def __init__(self, dim):
        assert dim>=1
        self._dim = dim
        self._bounds = np.zeros((2,dim))
        self._name = 'Ackley'
        print('Initialise Ackley function with ' + str(self._dim) + ' dimensions')
        
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
        candidates = candidates - np.pi
        
        if candidates.ndim==1:
            candidates = np.array([candidates])

        s1 = np.power(np.linalg.norm(candidates, axis=1), 2)
        s2 = np.sum(np.cos(2*np.pi*candidates), axis=1)
        obj_vals = -20.0 * np.exp(-0.2 * np.sqrt( (1.0/ self._dim) * s1)) - np.exp( (1.0/self._dim) * s2) + 20.0 + np.exp(1.0)

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
        b[0,:]*=-5
        b[1,:]*=10
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
