class BayesianNetworkBelief:
    def __init__(self, num_features, x, y):
        self.num_features = num_features

        self.covariance = np.zeros((num_features, num_features))

        for fi in num_features:
            for fj in num_features:
                covij = 0
                for xfi in x[fi]:
                    if xfi in x[fj]:
                        covij += y[fi] * y[fj]
                        

        
        self.covariance = np.cov(X)


        self.uncertainties = np.ones((num_parents, num_parents))
    
    def update(self, data):
        # compute empirical covariance matrix of features
        # estimate uncertainties of each term based on number of mutual observations of the variables involved
        # 