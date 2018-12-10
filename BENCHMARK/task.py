import numpy as np

class Task(object):
    def __init__(self, dim, fnc, ub, lb):
        self.dim = dim
        self.fnc = fnc
        self.lb = lb
        self.ub = ub

    def decode(self, rnvec):
        # nvars = rnvec[:self.dim]
        return self.lb + rnvec * (self.ub - self.lb)

    def encode(self, vec):
        return (vec - self.lb)/(self.ub - self.lb)