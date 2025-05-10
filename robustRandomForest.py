import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from random import randint
from distribution import Distribution

class RobustRandomForest:

    """
    Класс, реализующий алгоритм устойчивого случайного леса

    Атрибуты
    -------------
    estimators : list[(base_tree_class, pd.columns)], по умолчанию = []
        множество деревьев ансамбля

    estimators_samples :  list[(np.ndarray)], по умолчанию = []
        наблюдения, учавстующие в обучении i-го дерева

    X : pd.DataFrame
        набор наблюдений, на котором производилось обучение алгоритма

    y : pd.Series
        набор целевых значений, на котором производилось обучение алгоритма

    feature_importances : dict{feature: importance}, по умолчанию = {}
        значимость каждого, учавствующего в обучении признака
    """

    def __init__(self, regression=False, n_estimators=100, max_depth=None, max_features=1.0, n_jobs=-1, 
                 random_state=randint(0, 10000000), ccp_alpha=0.0, robustness=None, delta = 0.001):
        """
        Инициализация класса устойчивого случайного леса

        :param regression: bool, по умолчанию = Flase.
            флаг о регрессии.
        :param n_estimators: int, по умолчанию = 100.
            кол-во решающих деревьев.
        :param max_depth: int, по умолчанию = None.
            максимальная глубина решающего дерева.
        :param max_features: float, по умолчанию 1.0.
            флаг о регрессии.
        :param n_jobs: int, по умолчанию = -1.
            кол-во процессов, выделенных для параллельной работы.
        :param random_state: int, по умолчанию = None.
            случайное состояние.
        :param ccp_alpha: float, по умолчанию = 0.0.
            параметр пруннинга.
        :param robustness: str, по умолчанию = None.
            используемый алгоритм устойчивости.
        :param delta: float, по умолчанию 0.001.
            параметр сходимости.
        """
        self.regression = regression
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
        self.general_random = np.random.RandomState(self.random_state)
        self.delta = delta
        self.estimators_samples_ = []
        self.estimators_ = []
        self.robustness = robustness

 
    def _rsm_bootstrapping(self, X, y):
        """
        Построение бутстрэпированной выборки

        :param X: pd.DataFrame
            набор наблюдений, на котором будет производиться обучение алгоритма

        :param y: pd.Series
            набор целевых значений, на котором будет производиться обучение алгоритма

        :return X_b, y_b: pd.DataFrame, pd.Series
            бутстрэпированная выборка X_b и целевая переменная y_b
        """

        n_samples, n_features = X.shape
        if self.regression:
            max_features = self.max_features * n_features
        else:
            max_features = np.sqrt(n_features)

        sample_indexes = self.general_random.choice(n_samples, n_samples)
        features = self.general_random.choice(X.columns, round(max_features))
        X_b, y_b = X.iloc[sample_indexes][features], y.iloc[sample_indexes]

        self.estimators_samples_.append(sample_indexes)

        return X_b, y_b
    

    def __compute_rf_weights(self, X_train, X_new):
        """
        Вычисление весов Случайного леса

        :param X_train: pd.DataFrame
            набор наблюдений, на котором производилось обучение алгоритма

        :param X_new: pd.DataFrame
            набор наблюдений, на котором будет производиться прогноз алгоритма

        :return weights: np.ndarray, 
            веса случайного леса
        """

        n_train = X_train.shape[0]
        n_new = X_new.shape[0]
        weights = np.zeros((n_new, n_train))

        def compute(t, tree):

            # Получаем бутстрап-выборку из estimators_samples_
            bootstrap_sample = self.estimators_samples_[t]
            # Подсчитываем частоту каждого примера в выборке
            b_t = np.bincount(bootstrap_sample, minlength=n_train)
            
            # Терминальные узлы для нового и обучающего данных
            leaf_new = tree[0].apply(X_new[tree[0].feature_names_in_])
            leaf_train = tree[0].apply(X_train[tree[0].feature_names_in_])

            for i_new in range(n_new):
                mask = (leaf_train == leaf_new[i_new])
                denominator = np.sum(b_t * mask)

                if denominator == 0:
                    continue

                weights[i_new] += (b_t * mask) / denominator
            return

        # Параллельное вычисление весов
        Parallel(n_jobs=self.n_jobs, require='sharedmem')(delayed(compute)(t, tree) for t, tree in enumerate(self.estimators_))

        weights /= self.n_estimators
        return weights


    def _train_tree(self, X, y):
        """
        Обучение дерева решений

        :param X: pd.DataFrame
            набор наблюдений, на котором будет производиться обучение одного дерева

        :param y: pd.Series
            набор целевых значений, на котором будет производиться обучение одного дерева

        :return tree, columns: tree_base_class, pd.columns 
            Объект класса обученного дерева решений, набор призаков, участвующих в обучении
        """

        if self.regression:
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=self.random_state,
                                         ccp_alpha=self.ccp_alpha)
        else:
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          random_state=self.random_state,
                                          ccp_alpha=self.ccp_alpha)

        return tree.fit(X, y), X.columns
    
    
    def __compute_oob_weights(self, X_train):
        """
        Вычисление oob весов Сучайного леса

        :param X_train: pd.DataFrame
            набор наблюдений, на котором производилось обучение алгоритма

        :return oob_weights: np.ndarray, 
            OOB веса случайного леса
        """

        n = X_train.shape[0]

        def compute(j):

            oob_weights = np.zeros((n, n))# oob_weights[i, j] — вес i для j

            # Индексы деревьев, где j не участвовал (OOB)
            tree_indices = [t for t in range(self.n_estimators) if j not in self.estimators_samples_[t]]

            if not tree_indices:
                return
            
            for t in tree_indices:

                # Получение терминального узла для примера j в дереве t
                leaf_id = self.estimators_[t][0].apply(X_train.iloc[j:j + 1][self.estimators_[t][0].feature_names_in_])[0]

                # Индексы примеров в том же узле
                in_leaf = (self.estimators_[t][0].apply(X_train.iloc[j:j + 1][self.estimators_[t][0].feature_names_in_]) == leaf_id).flatten()

                # Бутстрап-веса для дерева t
                bootstrap_weights = np.bincount(self.estimators_samples_[t], minlength=n)

                # Вес примера i для j в дереве t
                numerator = bootstrap_weights * in_leaf
                denominator = np.sum(numerator)
                if denominator > 0:
                    oob_weights[:, j] = oob_weights[:, j] + (numerator / denominator)
            
            # Усреднение по всем деревьям в T_j
            oob_weights[:, j] /= len(tree_indices)

            return oob_weights
        
        # Параллельное вычисление весов
        self.__oob_weights = Parallel(n_jobs=self.n_jobs)(delayed(compute)(j) for j in range(n))

        oob_weights = self.__oob_weights[0]
        for i in range(1, len(self.__oob_weights)):
            oob_weights += self.__oob_weights[i]
        
        return oob_weights

    def __tukey_array(self, arr):
        """
        Построение массива проебразования Тьюки
        
        :param arr: np.ndarray
            массив, к которому применяется функция Tьюки

        :return tukey_arr: np.ndarray, 
            преобразованный массив 
        """

        tukey_arr = (1 - arr**2)**2
        tukey_arr[np.abs(arr) >= 1] = 0

        return tukey_arr


    def __oob_lambda(self, X_tarin, y_train, alpha):
        """
        Вычисление lambda для прогноза при robustness='lowess'
        
        :param X_tarin: pd.DataFrame
            набор наблюдений, на котором производилось обучение леса 

        :param y_train: pd.Series
            набор целевых значений, на котором производилось обучение леса 

        :param alpha: float
            параметр сходимости

        :return lambd: np.ndarray, 
            массив lambda
        """

        n = X_tarin.shape[0]
        oob_weights = self.__compute_oob_weights(X_tarin)
        oob_y = np.zeros(n)
        new_oob_y = oob_y.copy()

        # Инициализация начального приближения
        for j in range(n):
            oob_y[j] = np.dot(oob_weights[:, j], y_train)

        # Пока не превышено максимальное кол-во итераций
        for _ in range(10):

            oob_y = new_oob_y.copy()
            e = y_train - oob_y # Вычисляем ошибку
            m = np.median(abs(e))# Находим медиану ошибок
            lambd = self.__tukey_array(e / (alpha * m)) # Вычисляем массив lambda

            # Корректируем веса
            numerator = np.sum(lambd * oob_weights[:, j] * y_train) 
            denominator = np.sum(lambd * oob_weights[:, j])

            if denominator == 0:
                new_oob_y[j] = 0
            else:
                new_oob_y[j] = numerator / denominator

            # Если достигнута оптимальная точность возвращаем результат
            if np.sum((new_oob_y - oob_y) ** 2) / n < 0.0001:
                return lambd

        return lambd


    def fit(self, X_train, y_train, alpha = 10):
        """
        Обучение модели Robust Random Forest
        
        :param X_tarin: pd.DataFrame
            набор наблюдений, на котором будет производиться обучение леса 

        :param y_train: pd.Series
            набор целевых значений, на котором будет производиться обучение леса 

        :param alpha: float
            параметр сходимости

        :return: None
        """

        self.X = X_train
        self.y = y_train

        # Строим бутстрепированную выборку и параллельно обучаем на ней ансамбль деревьев 
        boot_data = (self._rsm_bootstrapping(X_train, y_train) for _ in range(self.n_estimators))
        train_trees = (delayed(self._train_tree)(X_b, y_b) for X_b, y_b in boot_data)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(train_trees)

        # Вычисляем значимость признаков
        self.__feature_importances()

        # Вычисляем lambda
        if self.robustness == 'lowess':
            self.oob_lambda = self.__oob_lambda(X_train, y_train, alpha)


    def predict(self, X_test):
        """
        Прогноз модели Robust Random Forest
        
        :param X_test: pd.DataFrame
            набор наблюдений, на котором будет производиться прогноз леса 

        :return forest_prediction: np.ndarray
            прогноз алгоритма
        """

        # Параллельно получаем прогнозы ансамбля
        prediction = (delayed(tree_i.predict)(X_test[tree_i_features]) for (tree_i, tree_i_features) in self.estimators_)
        trees_predictions = pd.DataFrame(Parallel(n_jobs=self.n_jobs)(prediction))

        # В зависимости от типа устойчивости выбираем соответствующий расчет прогноза
        if self.regression:
            if self.robustness == 'huber':
                forest_prediction = self.__weights_optimization_huber(trees_predictions)

            elif self.robustness == 'tukey':
                forest_prediction = self.__weights_optimization_tukey(trees_predictions)

            elif self.robustness == 'quantile':
                omega = self.__compute_rf_weights(self.X, X_test)
                forest_prediction = [np.sum(self.y * omega[i]) for i in range(X_test.shape[0])]

            elif self.robustness == 'lowess':
                omega = self.__compute_rf_weights(self.X, X_test)
                forest_prediction = [np.sum(self.oob_lambda * self.y * omega[i]) for i in range(X_test.shape[0])]

            else:
                forest_prediction = trees_predictions.mean(axis=0)
        else:
            forest_prediction = trees_predictions.mode(axis=0).iloc[0]

        return np.array(forest_prediction)


    def __feature_importances(self):

        """
        Вычисление значимости признаков
        
        :param: None

        :return: None
        """

        self.feature_importances_ = {}

        # Проходимся по всем деревьям в ансамбле
        for i in self.estimators_:

            # Получаем значимость и обучающие признаки
            importances = i[0].feature_importances_
            features = i[1].values

            # Добавляем в общий набор ансамбля
            for feature, importance in zip(features, importances):

                if feature not in self.feature_importances_.keys():
                    self.feature_importances_[feature] = importance
                else:
                    self.feature_importances_[feature] += importance

        # Нормируем значимость признаков
        for i in self.feature_importances_.keys():
            self.feature_importances_[i] /= self.n_estimators


    def __weights_optimization_huber(self, trees_predictions):
        """
        Вычисление прогноза при помощи псевдохуберовских потерь
        
        :param trees_predictions: pd.DataFrame
            прогнозы деревьев 

        :return forest_prediction: np.ndarray
            скорректированный прогноз алгоритма
        """
        
        training_responses = self.y
        # Инициализируем начальное приближение
        forest_prediction = trees_predictions.mean(axis=0)
        
        # Пока не превышено максимальное кол-во итераций
        for _ in range(10):

            new_forest_prediction = forest_prediction.copy()

            # Проходимся по каждому прогнозу деревьев
            for j in range(trees_predictions.shape[1]):

                numerator = 0
                denominator = 0
                # Нормируем начальные веса
                omega = np.ones(training_responses.shape[0])/training_responses.shape[0]

                # Проходимся по набор целевых значений, на котором производилось обучение алгоритма
                for i in range(training_responses.shape[0]):
                    numerator += omega[i] * training_responses[i] / np.sqrt(1 + ((new_forest_prediction[j] - training_responses[i])/self.delta)**2)
                    denominator += omega[i] /np.sqrt(1 + ((new_forest_prediction[j] - training_responses[i])/self.delta)**2)
                  
                # Корректируем прогноз
                if denominator != 0:
                    new_forest_prediction[j] = numerator/denominator

            # Если достигнута оптимальная точность возвращаем результат
            if max(abs(new_forest_prediction - trees_predictions.mean(axis=0))) > self.delta:
                return new_forest_prediction

            forest_prediction = new_forest_prediction

        return forest_prediction


    def __weights_optimization_tukey(self, trees_predictions):
        """
        Вычисление прогноза при помощи потерь Тьюки
        
        :param trees_predictions: pd.DataFrame
            прогнозы деревьев 

        :return forest_prediction: np.ndarray
            скорректированный прогноз алгоритма
        """

        training_responses = self.y
        # Инициализируем начальное приближение
        forest_prediction = trees_predictions.mean(axis=0)

        # Пока не превышено максимальное кол-во итераций
        for _ in range(10):

            new_forest_prediction = forest_prediction.copy()

            # Проходимся по каждому прогнозу деревьев
            for j in range(trees_predictions.shape[1]):

                numerator = 0
                denominator = 0
                # Нормируем начальные веса
                omega = np.ones(training_responses.shape[0])/training_responses.shape[0]

                # Проходимся по набор целевых значений, на котором производилось обучение алгоритма
                for i in range(training_responses.shape[0]):
                    omega[i] = omega[i] * max(1 - ((new_forest_prediction[j] - training_responses[i])/self.delta)**2, 0)
                    numerator += omega[i] * training_responses[i] 
                    denominator += omega[i] 

                # Корректируем прогноз 
                if denominator != 0:
                    new_forest_prediction[j] = numerator/denominator
            
            # Если достигнута оптимальная точность возвращаем результат
            if max(abs(new_forest_prediction - trees_predictions.mean(axis=0))) > self.delta:
                return new_forest_prediction
            
            forest_prediction = new_forest_prediction

        return forest_prediction
    

if __name__ == '__main__':

    dist = Distribution(ro1=0.1, ro2=1.0, _func=lambda x: x * np.sin(x))
    X_train, X_test, Y_train, Y_test = dist.distribution([-6, 6], 500)

    huber_rrf = RobustRandomForest(n_jobs=-1, regression=True, robustness='huber', delta=0.0001)
    huber_rrf.fit(X_train, Y_train)
    huber_rrf_pred = huber_rrf.predict(X_test)

    lowess_rrf = RobustRandomForest(n_jobs=-1, regression=True, robustness='lowess')
    lowess_rrf.fit(X_train, Y_train, alpha=20)
    lowess_rrf_pred = lowess_rrf.predict(X_test)

    quantile_rrf = RobustRandomForest(n_jobs=-1, regression=True, robustness='quantile')
    quantile_rrf.fit(X_train, Y_train)
    quantile_rrf_pred = quantile_rrf.predict(X_test)