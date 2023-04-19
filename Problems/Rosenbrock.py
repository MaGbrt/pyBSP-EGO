import numpy as np
from Problems.Problem import Problem

class Rosenbrock(Problem):
    """Class for the single-objective Rosenbrock problem.

    :param n_dvar: number of decision variable
    :type n_dvar: positive int, >1
    """

    #-------------__init__-------------#    
    def __init__(self, dim):
        assert dim>=1
        self._dim = dim
        self._bounds = np.zeros((2,dim))
        self._name = 'Rosenbrock'
        print('Initialise Rosenbrock function with ' + str(self._dim) + ' dimensions')
        
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
        if(self.is_feasible(candidates) == False):
            print('Problem with candidate ', candidates)
            print('Bounds are ', self._bounds)
        assert self.is_feasible(candidates)
        candidates = candidates - np.pi
        if candidates.ndim==1:
            candidates = np.array([candidates])
        obj_vals = np.sum(100*(candidates[:,0:candidates.shape[1]-1].__pow__(2)-candidates[:,1:]).__pow__(2) + (candidates[:,0:candidates.shape[1]-1]-1).__pow__(2), axis=1)
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
