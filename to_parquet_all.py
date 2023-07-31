import pandas as pd
import json


import numpy as np
from dask import dataframe as dd
from distributed import Client
from dask_jobqueue import SLURMCluster


cluster = SLURMCluster(

            processes=4,
            cores=8,
            memory='8GB',
            scheduler_options={'dashboard_address': ':8087'},
            walltime="01:00:00"
          )

cluster.scale(jobs=2)
client = Client(cluster, timeout="120s")

#cols = ['Street Name', 'Issuer Precinct', 'Issue Date', 'Violation Time', 'Violation County', "Summons# Number"]

path = "/d/hpc/projects/FRI/bigdata/data/NYTickets/"

#file_name = "2023_april"


names = ["2023_april", "2022_full", "2021", "2020", "2019", "2018"]

for f_name in names:
  full_path = path+f_name+".csv"
  df = dd.read_csv(full_path, assume_missing = True, dtype=str, header=0)
  df.to_parquet(f_name+".parquet")

