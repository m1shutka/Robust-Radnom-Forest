from sklearn.preprocessing import MaxAbsScaler
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

class Distribution():

    def __init__(self, alpha: float = 0.95, ro1: float = 0.05, ro2: float = 0.5, _func = None):

        self.alpha = alpha
        self.ro1 = ro1
        self.ro2 = ro2

        if _func != None:
            self.func = _func
        else:
            self.func = self.__default_func

        return


    def distribution(self, borders: list[float] = [-1, 1], N: int = 100, random_state: int = np.random.randint(99999999)):

        X = np.linspace(borders[0], borders[1], N)
        y = self.func(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=random_state)

        X_train = pd.DataFrame(data = {'X':X_train})
        X_test = pd.DataFrame({'X':X_test})

        Y_train = MaxAbsScaler().fit_transform(Y_train.reshape(-1, 1)).flatten()
        Y_train = pd.Series(Y_train)

        Y_test = MaxAbsScaler().fit_transform(Y_test.reshape(-1, 1)).flatten()
        Y_test = pd.Series(Y_test)

        c = np.sqrt(sum((Y_train - np.mean(Y_train))**2) / Y_train.shape[0])
        self.sigma1 = self.ro1 * c
        self.sigma2 = self.ro2

        for i in range(Y_train.shape[0]):
            if np.random.uniform(0, 1) < self.alpha:
                Y_train[i] += np.random.normal(0, self.sigma1)
            else:
                Y_train[i] += np.random.normal(0, self.sigma2)
                #Y_train[i] += np.random.randint(-2 * self.sigma2, 4 * self.sigma2)


        return X_train, X_test, Y_train, Y_test

    def __default_func(self, X):
        return X**2