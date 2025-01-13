import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from random import randint


class RobustRandomForest:

    def __init__(self, regression=False, n_estimators=100, max_depth=None,
                 max_features=1.0, n_jobs=-1, random_state=randint(0, 10000000), ccp_alpha=0.0, delta = 0.1):
        self.regression = regression
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
        self.trained_trees_info = []
        self.general_random = np.random.RandomState(self.random_state)
        self.delta = delta

 
    def _rsm_bootstrapping(self, X, y):
        n_samples, n_features = X.shape
        if self.regression:
            max_features = self.max_features * n_features
        else:
            max_features = np.sqrt(n_features)

        sample_indexes = self.general_random.choice(n_samples, n_samples)
        features = self.general_random.choice(X.columns, round(max_features))
        X_b, y_b = X.iloc[sample_indexes][features], y.iloc[sample_indexes]
        return X_b, y_b


    def _train_tree(self, X, y):
        if self.regression:
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=self.random_state,
                                         ccp_alpha=self.ccp_alpha)
        else:
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          random_state=self.random_state,
                                          ccp_alpha=self.ccp_alpha)

        return tree.fit(X, y), X.columns


    def fit(self, X, y):
        self.X = X
        self.y = y
        boot_data = (self._rsm_bootstrapping(X, y) for _ in range(self.n_estimators))
        train_trees = (delayed(self._train_tree)(X_b, y_b) for X_b, y_b in boot_data)
        self.trained_trees_info = Parallel(n_jobs=self.n_jobs)(train_trees)
        self.__feature_importances()


    def predict(self, samples, optimization=None):

        prediction = (delayed(tree_i.predict)(samples[tree_i_features])
                      for (tree_i, tree_i_features) in self.trained_trees_info)

        trees_predictions = pd.DataFrame(Parallel(n_jobs=self.n_jobs)(prediction))

        if self.regression:
            if optimization == 'huber':
                forest_prediction = self.__weights_optimization_huber(trees_predictions)
            elif optimization == 'tukey':
                forest_prediction = self.__weights_optimization_tukey(trees_predictions)
            elif optimization == 'huberV0':
                forest_prediction = self.__weights_optimization_huber2(trees_predictions)
            elif optimization == 'huber_test':
                forest_prediction = self.__weights_optimization_huber3(trees_predictions)
            else:
                forest_prediction = trees_predictions.mean(axis=0)
        else:
            forest_prediction = trees_predictions.mode(axis=0).iloc[0]

        return np.array(forest_prediction)


    def __feature_importances(self):

        self.feature_importances_ = {}

        for i in self.trained_trees_info:

            importances = i[0].feature_importances_
            features = i[1].values

            for feature, importance in zip(features, importances):

                if feature not in self.feature_importances_.keys():
                    self.feature_importances_[feature] = importance
                else:
                    self.feature_importances_[feature] += importance

        for i in self.feature_importances_.keys():
            self.feature_importances_[i] /= self.n_estimators


    def __weights_optimization_huber(self, trees_predictions):
        
        self.__training_responses = self.y
        forest_prediction = trees_predictions.mean(axis=0)
        
        while True:

            new_forest_prediction = forest_prediction.copy()

            for j in range(trees_predictions.shape[1]):

                upper = 0
                under = 0
                omega = np.ones(self.__training_responses.shape[0])/self.__training_responses.shape[0]

                for i in range(self.__training_responses.shape[0]):
                    upper += omega[i] * self.__training_responses[i] /np.sqrt(1 + ((new_forest_prediction[j] - self.__training_responses[i])/self.delta)**2)
                    under += omega[i] /np.sqrt(1 + ((new_forest_prediction[j] - self.__training_responses[i])/self.delta)**2)
                  
                new_forest_prediction[j] = upper/under


            #if np.sum((new_forest_prediction - forest_prediction) ** 2)/new_forest_prediction.shape[0] < 0.1:
            if max(abs(new_forest_prediction - trees_predictions.mean(axis=0))) > self.delta:
                return new_forest_prediction

            forest_prediction = new_forest_prediction

    def __weights_optimization_huber2(self, trees_predictions):
        
        self.__training_responses = self.predict(self.X)
        forest_prediction = trees_predictions.mean(axis=0)
        
        while True:

            new_forest_prediction = forest_prediction.copy()

            for j in range(trees_predictions.shape[1]):

                upper = 0
                under = 0
                omega = np.ones(self.__training_responses.shape[0])/self.__training_responses.shape[0]

                for i in range(self.__training_responses.shape[0]):
                    upper += omega[i] * self.__training_responses[i] /np.sqrt(1 + ((new_forest_prediction[j] - self.__training_responses[i])/self.delta)**2)
                    under += omega[i] /np.sqrt(1 + ((new_forest_prediction[j] - self.__training_responses[i])/self.delta)**2)
                  
                new_forest_prediction[j] = upper/under

            #if np.sum((new_forest_prediction - forest_prediction) ** 2)/new_forest_prediction.shape[0] < 0.1:
            if max(abs(new_forest_prediction - trees_predictions.mean(axis=0))) > self.delta:
                return new_forest_prediction
            
            forest_prediction = new_forest_prediction

    def __weights_optimization_tukey(self, trees_predictions):
        
        self.__training_responses = self.predict(self.X)
        forest_prediction = trees_predictions.mean(axis=0)
        
        while True:

            new_forest_prediction = forest_prediction.copy()

            for j in range(trees_predictions.shape[1]):

                upper = 0
                under = 0
                omega = np.ones(self.__training_responses.shape[0])/self.__training_responses.shape[0]

                for i in range(self.__training_responses.shape[0]):
                    omega[i] = omega[i] * max(1 - ((new_forest_prediction[j] - self.__training_responses[i])/self.delta)**2, 0)
                    upper += omega[i] * self.__training_responses[i] 
                    under += omega[i] 

                if under != 0:
                    new_forest_prediction[j] = upper/under
            
            #if np.sum((new_forest_prediction - forest_prediction) ** 2)/new_forest_prediction.shape[0] < 0.000001:
            if max(abs(new_forest_prediction - trees_predictions.mean(axis=0))) > self.delta:
                return new_forest_prediction

            forest_prediction = new_forest_prediction


    def __weights_optimization_huber3(self, trees_predictions):

        self.__training_responses = self.predict(self.X)
        forest_prediction = trees_predictions.mean(axis=0)

        while True: 
            new_forest_prediction = forest_prediction.copy()

            omega = np.array([np.ones(self.__training_responses.shape[0])/self.__training_responses.shape[0] for _ in range(new_forest_prediction.shape[0])]).T

            for j in range(new_forest_prediction.shape[0]):
                for i in range(self.__training_responses.shape[0]):
                    omega[i][j] = omega[i][j] / np.sqrt(1 + ((new_forest_prediction[j] - self.__training_responses[i])/self.delta)**2)
                    #print(f'{i*j}: omg = {omega[i][j]} (Y^ij - Yi)=({new_forest_prediction[j]} - {self.__training_responses[i]}) = {new_forest_prediction[j] - self.__training_responses[i]}')


            for j in range(new_forest_prediction.shape[0]):
                upper, under = 0, 0

                for i in range(self.__training_responses.shape[0]):
                    upper += omega[i][j] * self.__training_responses[i]
                    under += omega[i][j]   

                new_forest_prediction[j] = upper/under

            if np.sum((new_forest_prediction - forest_prediction) ** 2)/new_forest_prediction.shape[0] < 0.1:
            #if 2 * max(self.__training_responses) > self.delta:
                return new_forest_prediction

            forest_prediction = new_forest_prediction