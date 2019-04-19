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
<<<<<<< HEAD
        self.res = 8

        kernel = generate_rbfkern(2, 1.0, 1.5)
        xs = generate_grid(lb=-2.0, ub=2.0, res=self.res)
        ys = np.random.normal(size=(self.res * self.res, 1))

        print(xs, ys)
    
        self.true_res = 41
        model = GPy.models.GPRegression(xs, ys, kernel, noise_var=1e-10)
        x_disc = generate_grid(lb=-2.0, ub=2.0, res=self.true_res)
        print(x_disc)

        #model.plot()

        #self.truesignal = model.predict(x_disc)[0].reshape(true_res, true_res)
        #print(self.truesignal)


        #kernel = lambda x1, x2: math.exp(- np.linalg.norm(x1 - x2)**2 / 2.0)
        #training_inputs = generate_grid(lb=-2.0, ub=2.0, res=20)
        #K = 
        #for i in range(400):
        #    for j in range(400):
        #        kernel(training_inputs[i], training_inputs[j])


        self.truesignal = model.posterior_samples(x_disc, size=1).reshape(self.true_res, self.true_res)

        kernel = generate_rbfkern(2, 1.0, 1.5)
        xs = generate_grid(lb=-2.0, ub=2.0, res=self.res)
        ys = np.random.normal(size=(self.res * self.res, 1))
        model = GPy.models.GPRegression(xs, ys, kernel, noise_var=1e-10)
        x_disc = generate_grid(lb=-2.0, ub=2.0, res=self.true_res)
        print(x_disc)
        self.randsignal = model.posterior_samples(x_disc, size=1).reshape(self.true_res, self.true_res)
        print(self.truesignal)
=======
        self.res = 20

        self.kernel = generate_rbfkern(2, 1.0, 1.5)
        self.x_test = generate_grid(lb=-2.0, ub=2.0, res=self.res)
        self.draw = np.random.normal(size=(self.res * self.res, 1))

        #self.posteriorY = model.posterior_samples(x_samp, full_cov=True, size=1).reshape(self.res, self.res)
>>>>>>> 098165062136fb0c0d2750243a0ba6a0d3889115



    
    def load_prior_data(self, start_loc=[0, 0]):
        ''' inputs: none
            outputs:
                x_task: a numpy array of input-values for each sample for each function
                y_task: a numpy array of output-values for each sample from each function
                num_funcs: the number of state variables in the dataset
        '''
<<<<<<< HEAD
        n_sample = [100, 100, 1, 100]
=======
        n_sample = [100, 100, 1, 10]
>>>>>>> 098165062136fb0c0d2750243a0ba6a0d3889115
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

<<<<<<< HEAD
        xindex = int((x[0] + 2.0) / 4.0 * self.true_res-1)
        yindex = int((x[1] + 2.0) / 4.0 * self.true_res-1)

        #K_ss = self.kernel(self.x_test, x)
        #L = np.linalg.cholesky(K_ss + 1e-15*np.eye(len(K_ss)))
        #val = np.dot(L, self.draw)
        #print(x, self.res)
        #print(xindex, yindex)

        return self.truesignal[xindex, yindex]
=======
        #xindex = int((x[0] + 2.0) / 4 * self.res-1)
        #yindex = int((x[1] + 2.0) / 4 * self.res-1)

        K_ss = self.kernel(self.x_test, x)
        L = np.linalg.cholesky(K_ss + 1e-15*np.eye(len(K_ss)))
        val = np.dot(L, self.draw)

        #return self.posteriorY[xindex, yindex]
>>>>>>> 098165062136fb0c0d2750243a0ba6a0d3889115

    def depth_field(self, x):
        return -1 * x[0]
 
    def hardness_field(self, x):
        return (self.signal_field(x) - x[0]) * 2 #+ np.random.rand()*0.1

    def random_field(self, x):
<<<<<<< HEAD
        #return np.random.randn() #* 0.01
        xindex = int((x[0] + 2.0) / 4.0 * self.true_res-1)
        yindex = int((x[1] + 2.0) / 4.0 * self.true_res-1)


        return self.randsignal[xindex, yindex]
=======
        return 0.1 #+ np.random.randn() * 0.01
>>>>>>> 098165062136fb0c0d2750243a0ba6a0d3889115
