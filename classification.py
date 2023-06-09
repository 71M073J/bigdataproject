import pandas as pd
import numpy as np
from math import sin, cos, pi
import dask.dataframe as dd
from dask import array as da

import joblib
from time import time 
from dask import compute
from distributed import Client
import dask_jobqueue
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from dask_ml.metrics import accuracy_score
import math
from dask_ml.naive_bayes import GaussianNB
from dask_ml.model_selection import train_test_split
from dask_ml.metrics import accuracy_score
import xgboost as xgb


from dask_ml.linear_model import LinearRegression
from dask_ml.metrics import mean_squared_error



def xgboost_ml(X_train,y_train,X_test,y_test):
    train = xgb.dask.DaskDMatrix(client, da.from_array(X_train), da.from_array(y_train))
    test = xgb.dask.DaskDMatrix(client, da.from_array(X_test), da.from_array(y_test))
    
    s = time()
    
    params = {
        'verbosity': 3,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
    }
    output = xgb.dask.train(
        client,
        params,
        train,
        num_boost_round=4,
        verbose_eval=False,
        evals=[(test, "test")],
    )
    total_time = s - time()
    pred = xgb.dask.predict(client, output, test)
    #pred = pred.compute()
    acc = accuracy_score(y_test,pred)
    return acc,total_time % 60



def train_test_nb(*args):
    gnb = GaussianNB()
    s = time()
    x_train, x_test, y_train, y_test = args
    gnb.fit(x_train, y_train)
    total_time = time() - s
    
    preds = gnb.predict(x_test)
    preds = preds.compute()
    acc = accuracy_score(preds, y_test)
    
    return acc, total_time % 60

cluster = dask_jobqueue.SLURMCluster(
            processes=2,
            cores=8,
            memory='4GB',
            scheduler_options={'dashboard_address': ':8088'},
            walltime="00:15:00"
          )

cluster.scale(jobs=1)
client = Client(cluster,  timeout="120s")


def partial_fit(X,y):
    split = TimeSeriesSplit()
    total_time = 0
    accs, loss, times = [], [], []
    sgd = SGDClassifier(max_iter=150, tol=1e-6,loss='log')
    batches = split.split(X)
    
    for i,(t_idx, tst_idx) in enumerate(batches):
        X_train, X_test, y_train, y_test = X[t_idx], X[tst_idx], y[t_idx], y[tst_idx]

        with joblib.parallel_backend('dask'):
            if i == 0:
                s = time()
                sgd.fit(X_train,y_train)
                total_time = time() - s
            else:
                s = time()
                sgd.partial_fit(X_train,y_train)
                times.append(time()-s)
                
            pred = sgd.predict(X_test)
            accs.append(accuracy_score(y_test,pred))
            loss.append(log_loss(y_test,pred))
            
    return sum(accs)/len(accs),sum(loss)/len(loss),total_time % 60,sum(times)/len(times)


data = dd.read_parquet("agg_per_day.parquet", engine="pyarrow")
data = data.repartition(16)

print(data.columns)

def sin_(n):
    theta = 2 * pi * n
    return sin(theta)

def cos_(n):
    theta = 2 * pi * n
    return cos(theta)
#.timetuple().tm_yday

data['num_tix'] = data.apply(lambda row: int(row['num_tix']>9000), axis=1, meta=('num', 'bool'))


data['temp'] = data['temp'].astype(float)
data['humidity'] = data['humidity'].astype(float)
data['day_of_year'] = data['day_of_year'].astype(int)
data['day_of_year_sin'] = data.apply(lambda row: sin_(row['day_of_year']), axis=1, meta=('day', 'float'))
data['day_of_year_cos'] = data.apply(lambda row: cos_(row['day_of_year']), axis=1, meta=('day', 'float'))


data = data[(data['temp'].notnull()) & (data['humidity'].notnull())]




y = data['num_tix'].values.compute()

#data.drop('num_tix', axis=1)
X = data[['day_of_year_sin','day_of_year_cos', 'temp', 'humidity']].values.compute()
print("got ")
# X = X.rechunk('auto')
# y = y.rechunk('auto')
split = train_test_split(X, y, test_size=0.33, random_state=42)

print(train_test_nb(*split))


print(partial_fit(X, y))


#print(xgboost_ml(*split))
