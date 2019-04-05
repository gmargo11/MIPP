from data.SideInformationEnvironment import SideInformationEnvironment
from inference.JointGPModel import JointGPModel
from planning.ExploreExploitPathPlanner import ExploreExploitPathPlanner

import matplotlib.pyplot as plt



observed_feature = 2
loc = [0.0, 0.0]
points = [[-0.7, 0.0], [-0.7, 0.9],  [1.2, -1.3]]

env = SideInformationEnvironment(points)
planner = ExploreExploitPathPlanner()
model = JointGPModel()
model.load_environment(env)


alpha = 1.0

for i in range(50):

    # evaluate policy
    sampling_location = planner.policy(alpha, model.copy(), loc)
    
    # sample environment
    loc = sampling_location[0:2]
    observed_value = env.observe(sampling_location, observed_feature)
    
    # update model
    model.update(sampling_location, observed_value, observed_feature)

    # adjust explore-exploit parameter
    alpha = alpha * 0.99

    
    print(sampling_location)

model.display(feature=observed_feature, title='Signal Model')
plt.show()