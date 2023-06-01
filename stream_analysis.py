import time

import numpy as np
import pandas as pd
from streamz import Stream
from streamz.dataframe import DataFrame
import pyarrow.parquet as pq

def get_value(x, val):
    return x[val]

rmean = None
rmcount = None
def running_mean(state, x):
    if len(state) == 0:
        return x, 1
    else:
        rmean, rmcount, _ = state
    if rmean is None:
        rmean = x
        rmcount = 1
    else:
        rmcount += 1
        rmean = ((x * 1)/rmcount) + (rmean * (rmcount - 1))/rmcount
    return rmean, rmcount, x

#Wikipedia example for online variance, thanks wiki editors
def variance_update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def get_variance(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)

def get_tickets_today(state, x):
    if state == 0:
        return 1, state
    else:
        if x == state[1]:
            return state[0] + 1, x
        else:
            return 1, x

source = Stream()
# TODO running operations
#Running avg
#running std dev
#running_
tickets = pq.ParquetFile("parking.parquet")
#.map(lambda x: get_value(x, "Plate ID")) #za dobit en podatek iz datastreama
#.accumulate(running_mean, start=[]) #za dobit running mean, count
#.accumulate(variance_update, start=(0,0,0)).map(get_variance) # za dobit running mean, variance, samplevariance
#.map(lambda x: get_value(x, "Issue Date")).accumulate(get_tickets_today, start=0) # za dobit current daily tickets čeprav TODO ker ga napake(ne-trenuten datum) sesujejo
#.accumulate(lambda stt, x: (x if x > stt else stt)) #running max
#.accumulate(lambda stt, x: (x if x < stt else stt)) #running min
source.map(lambda x: get_value(x, "Issue Date")).accumulate(get_tickets_today, start=0).sink(print).start()
for chunk in tickets.iter_batches():
    chunk = chunk.to_pandas()
    for i in range(len(chunk)):
        source.emit(chunk.iloc[i])
        if i > 10000:
            quit()


#TODO stream clustering algo (lahko iz hw4?)




