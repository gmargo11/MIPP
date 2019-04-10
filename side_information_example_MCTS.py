from data.SideInformationEnvironmentRandomGP import SideInformationEnvironmentRandomGP
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



def simulateRun(env, plannerClass, modelClass, num_steps=50):

    observed_feature = 2
    loc = [1.0, 1.0]

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
    


def plot_MSE(env, plannerClass, modelClass, num_steps, num_trials, color):



    MSE_all = np.zeros((num_trials, num_steps))

    for i in range(num_trials):
        MSE, model = simulateRun(  env=env, 
                            plannerClass=plannerClass, 
                            modelClass=modelClass, 
                            num_steps=num_steps
                            )
        MSE_all[i] = MSE

    model.display(feature=2, title='Signal Model')

    plt.figure(1)
    plt.plot(range(num_steps), np.mean(MSE_all, axis=0), color)
    l1 = plt.fill_between(range(num_steps), np.mean(MSE_all, axis=0), np.mean(MSE_all, axis=0) + np.std(MSE_all, axis=0))
    l1.set_facecolors([[.5,.5,.8,.3]])
    l2 = plt.fill_between(range(num_steps), np.mean(MSE_all, axis=0), np.mean(MSE_all, axis=0) - np.std(MSE_all, axis=0))
    l2.set_facecolors([[.5,.5,.8,.3]])

    np.savetxt(str(type(env)) + " " + str(type(plannerClass)) + " " + str(type(modelClass)) + ".txt", MSE_all)
    #plt.plot(range(num_steps), np.mean(MSE_all, axis=0) + np.std(MSE_all, axis=0), 'g--')
    #plt.plot(range(num_steps), np.mean(MSE_all, axis=0) - np.std(MSE_all, axis=0), 'g--')



if __name__ == "__main__":
    points = [[-0.7, 0.0], [1.5, 0.9],  [1.2, -1.3]]

    plt.figure()

    plt.title('Model Mean Square Error')
    plt.xlabel('Time')
    plt.ylabel('MSE')


    plot_MSE(   env=SideInformationEnvironmentRandomGP(points), 
                plannerClass=GreedyPlanner, 
                modelClass=GaussianProcessBeliefModel, 
                num_steps=60,
                num_trials=3,
                color='r'
                )
    
    '''plot_MSE(   env=SideInformationEnvironment(points), 
                plannerClass=GreedyPlanner, 
                modelClass=GaussianProcessBeliefModel, 
                num_steps=50,
                num_trials=10,
                color='b'
                )
    '''
    plot_MSE(   env=SideInformationEnvironmentRandomGP(points), 
                plannerClass=GreedyPlanner, 
                modelClass=JointGPModel, 
                num_steps=60,
                num_trials=1,
                color='g'
                )
    '''
    plot_MSE(   env=SideInformationEnvironment(points), 
                plannerClass=GreedyPlanner, 
                modelClass=JointGPModel, 
                num_steps=20,
                num_trials=3,
                color='c'
                )
    '''
    plot_MSE(   env=SideInformationEnvironmentRandomGP(points), 
                plannerClass=GreedyPlanner, 
                modelClass=KnownCovarianceModel, 
                num_steps=60,
                num_trials=1,
                color='b'
                )

    plt.legend(['No Model', 'Fully Learned Model', 'Fully Expert Model']) #'Greedy IPP'])#, 'Side Information Augmented Random Path', 'Side Information Augmented Greedy IPP'])
    plt.show()