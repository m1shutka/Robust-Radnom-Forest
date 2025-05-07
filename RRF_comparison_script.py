import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from robustRandomForest import RobustRandomForest
from distribution import Distribution
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

ros = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
dims = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
alphas = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
noise_modules = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
n = 500

delta = 0.0001
sk_mae, huber_mae, tukey_mae, lowess_mae = [], [], [], []
sk_mse, huber_mse, tukey_mse, lowess_mse = [], [], [], []

for ro in tqdm(ros):

    mae, mse = [], []

    for i in range(n):
        
        dist = Distribution(ro1=ro, ro2=1.0, _func=lambda x: x * np.sin(x))
        X_train, X_test, Y_train, Y_test = dist.distribution([-6, 6], 500)

        huber_rrf = RobustRandomForest(n_jobs=-1, regression = True, robustness='huber', delta=delta)
        huber_rrf.fit(X_train, Y_train)
        huber_rrf_pred = huber_rrf.predict(X_test)
        mae.append(mean_absolute_error(huber_rrf_pred, Y_test))
        mse.append(mean_squared_error(huber_rrf_pred, Y_test))

        tukey_rrf = RobustRandomForest(n_jobs=-1, regression = True, robustness='tukey', delta=delta)
        tukey_rrf.fit(X_train, Y_train)
        tukey_rrf_pred = tukey_rrf.predict(X_test)
        mae.append(mean_absolute_error(tukey_rrf_pred, Y_test))
        mse.append(mean_squared_error(tukey_rrf_pred, Y_test))

        lowess_rrf = RobustRandomForest(n_jobs=-1, regression = True, robustness='lowess')
        lowess_rrf.fit(X_train, Y_train, alpha=20)
        lowess_rrf_pred = lowess_rrf.predict(X_test)
        mae.append(mean_absolute_error(lowess_rrf_pred, Y_test))
        mse.append(mean_squared_error(lowess_rrf_pred, Y_test))
        
        sk_rf = RandomForestRegressor(n_jobs=-1)
        sk_rf.fit(X_train, Y_train)
        sk_rf_pred = sk_rf.predict(X_test)
        mae.append(mean_absolute_error(sk_rf_pred, Y_test))
        mse.append(mean_squared_error(sk_rf_pred, Y_test))
    
    huber_mae.append(np.mean(mae[0: n*4:4]))
    tukey_mae.append(np.mean(mae[1: n*4:4]))
    lowess_mae.append(np.mean(mae[2: n*4:4]))
    sk_mae.append(np.mean(mae[3: n*4:4]))

    huber_mse.append(np.mean(mse[0: n*4:4]))
    tukey_mse.append(np.mean(mse[1: n*4:4]))
    lowess_mse.append(np.mean(mse[2: n*4:4]))
    sk_mse.append(np.mean(mse[3: n*4:4]))

    ros_range_data = {'ros': ros, 
      'huber_mae': huber_mae, 
      'tukey_mae': tukey_mae,
      'lowess_mae': lowess_mae,
      'sk_mae': sk_mae,
      'huber_mse': huber_mse,
      'tukey_mse': tukey_mse,
      'lowess_mse': lowess_mse,
      'sk_mse': sk_mse}

df = pd.DataFrame(ros_range_data)
df.to_csv('ros_range_data.csv')

sk_mae, huber_mae, tukey_mae, lowess_mae = [], [], [], []
sk_mse, huber_mse, tukey_mse, lowess_mse = [], [], [], []

for dim in tqdm(dims):

    mae, mse = [], []

    for i in range(n):
        
        dist = Distribution(ro1=0.1, ro2=1.0, _func=lambda x: x * np.sin(x))
        X_train, X_test, Y_train, Y_test = dist.distribution([-6, 6], dim)

        huber_rrf = RobustRandomForest(n_jobs=-1, regression = True, robustness='huber', delta=delta)
        huber_rrf.fit(X_train, Y_train)
        huber_rrf_pred = huber_rrf.predict(X_test)
        mae.append(mean_absolute_error(huber_rrf_pred, Y_test))
        mse.append(mean_squared_error(huber_rrf_pred, Y_test))

        tukey_rrf = RobustRandomForest(n_jobs=-1, regression = True, robustness='tukey', delta=delta)
        tukey_rrf.fit(X_train, Y_train)
        tukey_rrf_pred = tukey_rrf.predict(X_test)
        mae.append(mean_absolute_error(tukey_rrf_pred, Y_test))
        mse.append(mean_squared_error(tukey_rrf_pred, Y_test))

        lowess_rrf = RobustRandomForest(n_jobs=-1, regression = True, robustness='lowess')
        lowess_rrf.fit(X_train, Y_train, alpha=20)
        lowess_rrf_pred = lowess_rrf.predict(X_test)
        mae.append(mean_absolute_error(lowess_rrf_pred, Y_test))
        mse.append(mean_squared_error(lowess_rrf_pred, Y_test))
        
        sk_rf = RandomForestRegressor(n_jobs=-1)
        sk_rf.fit(X_train, Y_train)
        sk_rf_pred = sk_rf.predict(X_test)
        mae.append(mean_absolute_error(sk_rf_pred, Y_test))
        mse.append(mean_squared_error(sk_rf_pred, Y_test))
    
    huber_mae.append(np.mean(mae[0: n*4:4]))
    tukey_mae.append(np.mean(mae[1: n*4:4]))
    lowess_mae.append(np.mean(mae[2: n*4:4]))
    sk_mae.append(np.mean(mae[3: n*4:4]))

    huber_mse.append(np.mean(mse[0: n*4:4]))
    tukey_mse.append(np.mean(mse[1: n*4:4]))
    lowess_mse.append(np.mean(mse[2: n*4:4]))
    sk_mse.append(np.mean(mse[3: n*4:4]))

dims_range_data = {'dims': dims, 
      'huber_mae': huber_mae, 
      'tukey_mae': tukey_mae,
      'lowess_mae': lowess_mae,
      'sk_mae': sk_mae,
      'huber_mse': huber_mse,
      'tukey_mse': tukey_mse,
      'lowess_mse': lowess_mse,
      'sk_mse': sk_mse}

df = pd.DataFrame(dims_range_data)
df.to_csv('dims_range_data.csv')

sk_mae, huber_mae, tukey_mae, lowess_mae = [], [], [], []
sk_mse, huber_mse, tukey_mse, lowess_mse = [], [], [], []

for alpha in tqdm(alphas):

    mae, mse = [], []

    for i in range(n):
        
        dist = Distribution(alpha=alpha, ro2=1.0, _func=lambda x: x * np.sin(x))
        X_train, X_test, Y_train, Y_test = dist.distribution([-6, 6], 500)

        huber_rrf = RobustRandomForest(n_jobs=-1, regression = True, robustness='huber', delta=delta)
        huber_rrf.fit(X_train, Y_train)
        huber_rrf_pred = huber_rrf.predict(X_test)
        mae.append(mean_absolute_error(huber_rrf_pred, Y_test))
        mse.append(mean_squared_error(huber_rrf_pred, Y_test))

        tukey_rrf = RobustRandomForest(n_jobs=-1, regression = True, robustness='tukey', delta=delta)
        tukey_rrf.fit(X_train, Y_train)
        tukey_rrf_pred = tukey_rrf.predict(X_test)
        mae.append(mean_absolute_error(tukey_rrf_pred, Y_test))
        mse.append(mean_squared_error(tukey_rrf_pred, Y_test))

        lowess_rrf = RobustRandomForest(n_jobs=-1, regression = True, robustness='lowess')
        lowess_rrf.fit(X_train, Y_train, alpha=20)
        lowess_rrf_pred = lowess_rrf.predict(X_test)
        mae.append(mean_absolute_error(lowess_rrf_pred, Y_test))
        mse.append(mean_squared_error(lowess_rrf_pred, Y_test))
        
        sk_rf = RandomForestRegressor(n_jobs=-1)
        sk_rf.fit(X_train, Y_train)
        sk_rf_pred = sk_rf.predict(X_test)
        mae.append(mean_absolute_error(sk_rf_pred, Y_test))
        mse.append(mean_squared_error(sk_rf_pred, Y_test))
    
    huber_mae.append(np.mean(mae[0: n*4:4]))
    tukey_mae.append(np.mean(mae[1: n*4:4]))
    lowess_mae.append(np.mean(mae[2: n*4:4]))
    sk_mae.append(np.mean(mae[3: n*4:4]))

    huber_mse.append(np.mean(mse[0: n*4:4]))
    tukey_mse.append(np.mean(mse[1: n*4:4]))
    lowess_mse.append(np.mean(mse[2: n*4:4]))
    sk_mse.append(np.mean(mse[3: n*4:4]))

alphas_range_data = {'alphas': alphas, 
      'huber_mae': huber_mae, 
      'tukey_mae': tukey_mae,
      'lowess_mae': lowess_mae,
      'sk_mae': sk_mae,
      'huber_mse': huber_mse,
      'tukey_mse': tukey_mse,
      'lowess_mse': lowess_mse,
      'sk_mse': sk_mse}

df = pd.DataFrame(alphas_range_data)
df.to_csv('alphas_range_data.csv')

sk_mae, huber_mae, tukey_mae, lowess_mae = [], [], [], []
sk_mse, huber_mse, tukey_mse, lowess_mse = [], [], [], []

for noise_module in tqdm(noise_modules):

    mae, mse = [], []

    for i in range(n):
        
        dist = Distribution(alpha=0.9, ro2=noise_module, _func=lambda x: x * np.sin(x))
        X_train, X_test, Y_train, Y_test = dist.distribution([-6, 6], 500)

        huber_rrf = RobustRandomForest(n_jobs=-1, regression = True, robustness='huber', delta=delta)
        huber_rrf.fit(X_train, Y_train)
        huber_rrf_pred = huber_rrf.predict(X_test)
        mae.append(mean_absolute_error(huber_rrf_pred, Y_test))
        mse.append(mean_squared_error(huber_rrf_pred, Y_test))

        tukey_rrf = RobustRandomForest(n_jobs=-1, regression = True, robustness='tukey', delta=delta)
        tukey_rrf.fit(X_train, Y_train)
        tukey_rrf_pred = tukey_rrf.predict(X_test)
        mae.append(mean_absolute_error(tukey_rrf_pred, Y_test))
        mse.append(mean_squared_error(tukey_rrf_pred, Y_test))

        lowess_rrf = RobustRandomForest(n_jobs=-1, regression = True, robustness='lowess')
        lowess_rrf.fit(X_train, Y_train, alpha=20)
        lowess_rrf_pred = lowess_rrf.predict(X_test)
        mae.append(mean_absolute_error(lowess_rrf_pred, Y_test))
        mse.append(mean_squared_error(lowess_rrf_pred, Y_test))
        
        sk_rf = RandomForestRegressor(n_jobs=-1)
        sk_rf.fit(X_train, Y_train)
        sk_rf_pred = sk_rf.predict(X_test)
        mae.append(mean_absolute_error(sk_rf_pred, Y_test))
        mse.append(mean_squared_error(sk_rf_pred, Y_test))
    
    huber_mae.append(np.mean(mae[0: n*4:4]))
    tukey_mae.append(np.mean(mae[1: n*4:4]))
    lowess_mae.append(np.mean(mae[2: n*4:4]))
    sk_mae.append(np.mean(mae[3: n*4:4]))

    huber_mse.append(np.mean(mse[0: n*4:4]))
    tukey_mse.append(np.mean(mse[1: n*4:4]))
    lowess_mse.append(np.mean(mse[2: n*4:4]))
    sk_mse.append(np.mean(mse[3: n*4:4]))

noise_modules_range_data = {'noise_modules': noise_modules, 
      'huber_mae': huber_mae, 
      'tukey_mae': tukey_mae,
      'lowess_mae': lowess_mae,
      'sk_mae': sk_mae,
      'huber_mse': huber_mse,
      'tukey_mse': tukey_mse,
      'lowess_mse': lowess_mse,
      'sk_mse': sk_mse}

df = pd.DataFrame(noise_modules_range_data)
df.to_csv('noise_modules_range_data.csv')