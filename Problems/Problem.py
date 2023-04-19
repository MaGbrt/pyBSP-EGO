import numpy as np
from abc import ABC, abstractmethod


#------------------------------------------------#
#-------------abstract class Problem-------------#
#------------------------------------------------#
class Problem(ABC):
    """Abstract class for single objective real-valued optimization problems.
    :param dim: dimension of the input space
    :type dim: positive int, not zero
    """

    #-------------__init__-------------#
    @abstractmethod
    def __init__(self, dim, cec=None):
        assert type(dim)==int
        assert dim>0 
        self._dim = dim
        self._name = 'str'
        self._bounds = np.ones(shape=(2, dim), dtype=float, order='C')
        self.bounds[0,:] *= 0
        self.bounds[1,:] *= 1
 
    #-------------__del__-------------#
    @abstractmethod
    def __del__(self):
        del self._dim

        
    @abstractmethod
    def set_bounds(self, bounds):
        assert type(bounds) == np.ndarray, 'Bounds should be of type numpy.ndarray'
            
    
    @abstractmethod
    def evaluate(self, candidates):
        assert self.is_feasible(candidates)==True

    #-------------is_feasible-------------#
    @abstractmethod
    def is_feasible(self, candidates):
        res=False
        if type(candidates) is np.ndarray:
            if candidates.ndim==1:
                res=(candidates.size==self._dim)
            else:
                res=(candidates.shape[1]==self._dim)
        else :
            print('This function accepts only numpy ndarrays.')
        return res
