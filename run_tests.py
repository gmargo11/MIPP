from data.SideInformationEnvironmentRandomGP import SideInformationEnvironmentRandomGP
from data.SideInformationEnvironmentRandomIndependentGP import SideInformationEnvironmentRandomIndependentGP
from inference.DiscreteGaussianBeliefModel import DiscreteGaussianBeliefModel
from inference.GaussianProcessBeliefModel import GaussianProcessBeliefModel
from inference.JointGPModel import JointGPModel
from inference.KnownCovarianceModel import KnownCovarianceModel
from planning.MCTSPlanner import MCTSPlanner
from planning.GreedyPlanner import GreedyPlanner
from planning.BanditPlanner import BanditPlanner
from planning.RandomPlanner import RandomPlanner
from planning.ReplanningBanditPlanner import ReplanningBanditPlanner

import matplotlib.pyplot as plt
import numpy as np



def simulateRun(env, plannerClass, modelClass, seed, num_steps=50, loc=[0.0, 0.0]):

    observed_feature = 2
    env.randomize(seed)

    planner = plannerClass()
    model = modelClass()
    model.load_environment(env, loc)


    alpha = 1.0
    MSE = []

    for i in range(num_steps):

        # evaluate policy
        sampling_location = planner.policy(alpha, model, loc)
        
        # sample environment
        loc = sampling_location[0:2]
        observed_value = env.observe(sampling_location, observed_feature)
        
        # update model
        model.update(sampling_location, observed_value, observed_feature)

        # adjust explore-exploit parameter
        alpha = alpha * 0.99
        MSE.append(model.evaluate_MSE(env.func[2]))

        
        print('Sampled at: ', sampling_location)


    #model.display(feature=observed_feature, title='Signal Model')
    #plt.figure(1)
    #plt.plot(range(len(MSE)), MSE)

    return MSE, model
    


def plot_MSE(env, plannerClass, modelClass, num_steps, num_trials, color, loc_list):


    MSE_all = np.zeros((num_trials, num_steps))

    for i in range(num_trials):
        MSE, model = simulateRun(  env=env, 
                            plannerClass=plannerClass, 
                            modelClass=modelClass, 
                            seed=i,
                            num_steps=num_steps,
                            loc=loc_list[i]
                            )
        MSE_all[i] = MSE

    model.display(feature=2, title='Signal Model')

    plt.figure(1)
    plt.plot(range(num_steps), np.mean(MSE_all, axis=0), color)
    l1 = plt.fill_between(range(num_steps), np.mean(MSE_all, axis=0), np.mean(MSE_all, axis=0) + np.std(MSE_all, axis=0))
    l1.set_facecolors([[.5,.5,.8,.3]])
    l2 = plt.fill_between(range(num_steps), np.mean(MSE_all, axis=0), np.mean(MSE_all, axis=0) - np.std(MSE_all, axis=0))
    l2.set_facecolors([[.5,.5,.8,.3]])

    #np.savetxt(str(type(env)) + " " + str(type(plannerClass)) + " " + str(type(modelClass)) + ".txt", MSE_all)
    #plt.plot(range(num_steps), np.mean(MSE_all, axis=0) + np.std(MSE_all, axis=0), 'g--')
    #plt.plot(range(num_steps), np.mean(MSE_all, axis=0) - np.std(MSE_all, axis=0), 'g--')



if __name__ == "__main__":
    points = [[-0.7, 0.0], [1.5, 0.9],  [1.2, -1.3]]

    plt.figure()

    plt.title('Temperature Prediction Mean Square Error')
    plt.xlabel('Time (min)')
    plt.ylabel('MSE')

    env = SideInformationEnvironmentRandomGP(points)
    #env = SideInformationEnvironmentRandomIndependentGP(points)

    num_trials = 12
    num_steps = 80

    starting_locs = np.round(np.random.random(size=(num_trials, 2)), 1)

    '''
    plot_MSE(   env=env, 
                plannerClass=RandomPlanner, 
                modelClass=GaussianProcessBeliefModel, 
                num_steps=num_steps,
                num_trials=num_trials,
                color='r',
                loc_list=starting_locs
                )
    '''
    plot_MSE(   env=env, 
                plannerClass=GreedyPlanner, 
                modelClass=GaussianProcessBeliefModel, 
                num_steps=num_steps,
                num_trials=num_trials,
                color='b',
                loc_list=starting_locs
                )
    
    plot_MSE(   env=env, 
                plannerClass=GreedyPlanner, 
                modelClass=JointGPModel, 
                num_steps=num_steps,
                num_trials=num_trials,
                color='g',
                loc_list=starting_locs
                )
    '''
    plot_MSE(   env=env, 
                plannerClass=GreedyPlanner, 
                modelClass=JointGPModel, 
                num_steps=num_steps,
                num_trials=num_trials,
                color='c',
                loc_list=starting_locs
                )
    '''
    plot_MSE(   env=env, 
                plannerClass=GreedyPlanner, 
                modelClass=KnownCovarianceModel, 
                num_steps=num_steps,
                num_trials=num_trials,
                color='c',
                loc_list=starting_locs
                )
    

    plt.legend(['S-IPP', 'M-IPP with Parameter Learning', 'Fully Expert Model']) #'Greedy IPP'])#, 'Side Information Augmented Random Path', 'Side Information Augmented Greedy IPP'])
    #plt.legend(['S-IPP, Randomly Exploring', 'S-IPP, Greedy Information Gain', 'M-IPP, Greedy Information Gain']) #'Greedy IPP'])#, 'Side Information Augmented Random Path', 'Side Information Augmented Greedy IPP'])
    plt.show()