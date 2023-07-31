import pandas as pd
import numpy as np

import dask.dataframe as dd
from dask import array as da


from time import time 
from dask import compute
from distributed import Client
import dask_jobqueue

import math

from dask_ml.model_selection import train_test_split

import xgboost


from dask_ml.linear_model import LinearRegression
from dask_ml.metrics import mean_squared_error


# cluster = dask_jobqueue.SLURMCluster(
#             processes=2,
#             cores=8,
#             memory='16GB',
#             scheduler_options={'dashboard_address': ':8088'},
#             walltime="00:15:00"
#           )

# cluster.scale(jobs=4)
# client = Client(cluster, timeout="120s")


# ML TASK: Predict days with a high number of tickets

# group dataset by date 
# weather data is the same for everything - take first
# business data - same for everything - take first
# school - same for everything 
# events - also same?
# take first for all 

names = ["2023_april", "2022_full", "2021", "2020", "2019", "2018"]

files = ["with_weather_dask" + x + ".parquet" for x in names]


data = dd.read_parquet(files, engine="pyarrow").set_index('Summons Number')
data = data.repartition(16)
data['datetime_issue'] = data['datetime_issue'].map_partitions(pd.to_datetime, format="%Y-%m-%d %H:%M:%S", meta=('datetime_issue', 'datetime64[ns]'))
#data['datetime_issue'] =  dd.to_datetime(data['datetime_issue'])
data['date'] = data['datetime_issue'].dt.date.astype(str).compute()
data['day_of_year'] = data['datetime_issue'].dt.strftime('%j')
data['num_tix'] = 1

# df.groupby('name').agg({'x': ['mean', 'std'], 'y': ['mean', 'count']}).compute().head()

data = data.groupby('date').agg({'num_tix': 'sum',
'temp': 'max',
'humidity': 'max',
'snowdepth': 'max',
'conditions': 'first',
'description': 'first',
'day_of_year': 'first'
}).compute()

# print(data_g['num_tix'].mean())

# data_g = data_g.merge(data, how="left", on='date')
# data_g = data_g.repatition(16)


# del data 


data.to_parquet("agg_per_day.parquet")


def xgboost_ml(X_train,X_test,y_train,y_test):
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


data = dd.from_pandas(data, npartitions=32)



X = data[[i for i in data.columns if i != 'num_tix']].values.compute()
y = data['num_tix'].values.compute()

# X = da.from_array(X).compute()
# y = da.from_array(y).compute()


# X = X.rechunk('auto')
# y = y.rechunk('auto')


X_train, X_test, y_train, y_test = train_test_split(
  X,
  y
)


try:
    print(xgboost_ml(X_train, X_test, y_train, y_test))
except Exception as e:
    print("xgberror")
    print(e)



def baseline_m(X_train, y_train, X_test, y_test):
    s = time()
    baseline = LinearRegression()
    baseline.fit(X_train[:, []], y_train)

    y_baseline = baseline.predict(X_test[:, []])
    mse_baseline = mean_squared_error(y_test, y_baseline)
    print(mse_baseline)
    total_time = time() - S
    print(total_time / 60)

try:
    baseline_m(X_train, y_train, X_test, y_test)
except Exception as e:
    print("lin reg error")
    print(e)