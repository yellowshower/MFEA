import numpy as np
import scipy.optimize as op

class Individual(object):

    def __init__(self, D_multitask, tasks):
        self.dim = D_multitask
        self.tasks = tasks
        self.no_of_tasks = len(tasks)

        self.rnvec = np.random.uniform(size=D_multitask)
        self.scalar_fitness = None
        self.skill_factor = None

    def evaluate(self, p_il,method):
        if self.skill_factor == None:
            raise ValueError("skill factor not set")
        else:
            task = self.tasks[self.skill_factor]
            nvars = self.rnvec[:task.dim]
            vars = task.decode(nvars)
            if np.random.uniform() <= p_il:
                res = op.minimize(task.fnc, vars, method=method)
                x = res.x
                mvars = task.encode(x)
                # m_nvars = nvars
                # m_nvars[m_nvars < 0] = 0
                # m_nvars[m_nvars > 1] = 1
                mvars[mvars<0]=0
                mvars[mvars>1]=1
                # if (m_nvars != nvars).shape[0] != 0:
                #     nvars = m_nvars
                x = task.decode(mvars)
                objective = task.fnc(x)
                self.rnvec[: task.dim] = mvars
                funcCount = res.nfev
            else:
                objective = task.fnc(vars)
                funcCount = 1
        return self.skill_factor, objective, funcCount
