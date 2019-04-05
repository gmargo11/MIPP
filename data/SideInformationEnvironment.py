from data.Environment import Environment
import numpy as np
import math
from inference.GP_helpers import generate_grid

class SideInformationEnvironment(Environment):
    def __init__(self, points):
        self.points = points
        self.func = [self.hardness_field, self.depth_field, self.signal_field, self.random_field]
        self.func_names = ['hardness field', 'depth field', 'signal field', 'random field']
    
    def load_prior_data(self):
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
                x_task[i] = np.array([np.random.rand(n_sample[i]) * -1, np.random.rand(n_sample[i]) * -1]).T
            else:
                x_task[i] = generate_grid(-2.0, 2.0, int(math.sqrt(n_sample[i])))
            y_task[i] = np.array([self.func[i](xp) for xp in x_task[i]])
            
        return x_task, y_task, num_funcs
    
    def observe(self, x, feature):
        return self.func[feature](x)


    def signal_field(self, x):
        #points = [[-0.5, 0.5], [0.2, 0.2], [0.3, -0.6], [0.9, 0.3], [-0.1, -0.8]]
        strength = 0
        for point in self.points:
            dist = math.sqrt((x[0] - point[0])**2 + (x[1] - point[1])**2)
            strength += math.exp(-dist*3) * 2
        return strength

    def depth_field(self, x):
        return -1 * x[0]
 
    def hardness_field(self, x):
        return (self.signal_field(x) - x[0]) * 2 #+ np.random.rand()*0.1

    def random_field(self, x):
        return 0.1 + np.random.randn() * 0.01
