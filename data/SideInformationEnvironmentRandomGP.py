from data.Environment import Environment
import numpy as np
import math
from inference.GP_helpers import generate_grid, generate_rbfkern
import GPy

class SideInformationEnvironmentRandomGP(Environment):
    def __init__(self, points):
        self.points = points
        self.func = [self.hardness_field, self.depth_field, self.signal_field, self.random_field]
        self.func_names = ['hardness field', 'depth field', 'signal field', 'random field']
        self.res = 50

        kernel = generate_rbfkern(2, 1.0, 1.5)
        self.model = GPy.models.GPRegression(np.array(generate_grid(lb=-2.0, ub=2.0, res=self.res)), np.random.randn(self.res * self.res).reshape(-1, 1), kernel)
        self.signal_grid = self.model.predict(np.array(generate_grid(lb=-2.0, ub=2.0, res=50)))[0].reshape(50, 50)

        print(self.signal_grid)

        #x_samp = generate_grid(lb=-2.0, ub=2.0, res=self.res)
        #self.posteriorY = model.posterior_samples(x_samp, size=1).reshape(self.res, self.res)



    
    def load_prior_data(self, start_loc=[0, 0]):
        ''' inputs: none
            outputs:
                x_task: a numpy array of input-values for each sample for each function
                y_task: a numpy array of output-values for each sample from each function
                num_funcs: the number of state variables in the dataset
        '''
        n_sample = [100, 100, 1, 10]
        num_funcs = len(self.func)
        x_task = np.array([None for i in range(num_funcs)])
        y_task = np.array([None for i in range(num_funcs)])

        ### generate training data from functions
        for i in range(num_funcs):
            if i == 2: 
                #x_task[i] = np.array([np.random.rand(n_sample[i]) * -1, np.random.rand(n_sample[i]) * -1]).T
                x_task[i] = np.array([[start_loc[0]], [start_loc[1]]]).T
            else:
                x_task[i] = generate_grid(-2.0, 2.0, int(math.sqrt(n_sample[i])))
            y_task[i] = np.array([self.func[i](xp) for xp in x_task[i]])
            
        return x_task, y_task, num_funcs
    
    def observe(self, x, feature):
        return self.func[feature](x)


    def signal_field(self, x):
        x_index = int((x[0] + 2.0) / 4.0 * 50 - 1)
        y_index = int((x[0] + 2.0) / 4.0 * 50 - 1)

        return self.signal_grid[x_index][y_index]

    def depth_field(self, x):
        return -1 * x[0]
 
    def hardness_field(self, x):
        return (self.signal_field(x) - x[0]) * 2 #+ np.random.rand()*0.1

    def random_field(self, x):
        return 0.1 #+ np.random.randn() * 0.01
