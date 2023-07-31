import pandas as pd
import json


import numpy as np
from dask import dataframe as dd
from distributed import Client
from dask_jobqueue import SLURMCluster

schools = pd.read_csv('./data/school_locations.csv')
def street_from_address(x):
    # make sure the addresses are written correctly
    a = x.split(" ")[1:]
    return ' '.join(a)
    
def fix_street_abbreviation(x):
    abb = str(x).split()[-1]

    if abb in street_suffix:
        x = x.replace(abb, street_suffix[abb][0])
    return x

cluster = SLURMCluster(

            processes=4,
            cores=8,
            memory='8GB',
            scheduler_options={'dashboard_address': ':8087'},
            walltime="01:00:00"
          )

cluster.scale(jobs=2)
client = Client(cluster, timeout="120s")


#client = Client(cluster, timeout="120s")
cols = ['Street Name', 'Issuer Precinct', 'Issue Date', 'Violation Time']



path = "/d/hpc/projects/FRI/bigdata/data/NYTickets/"

names = ["2023_april", "2022_full", "2021", "2020", "2019", "2018"]

for f_name in names:
    
    full_path = path+f_name+".csv"
    df = dd.read_csv(full_path, assume_missing = True, usecols=cols, dtype={
    'Issuer Precinct': np.float64, })
  
    df = df.repartition(npartitions=16)
	
    df['Street Name'] = df['Street Name'].astype(str)

    f = open('data/street-suffix-abbreviations.json')
    street_suffix = json.load(f)
    df["Street Name"] = df.apply(lambda row: fix_street_abbreviation(row['Street Name']), axis = 1, meta=('Street Name', 'str'))

    schools['Street Name'] = schools.apply(lambda row: street_from_address(row['primary_address_line_1']), axis = 1)

    # extract relevant categories
    schools = schools[["Street Name", "Location_Category_Description", "Police_precinct"]]

    schools = schools[["Street Name", "Location_Category_Description", "Police_precinct"]]

    # note we are choosing to rename to issuer precinct. 
    # we could also join via Violation Precinct - it would give us a different picture
    schools = schools.rename(columns={"Police_precinct": "Issuer Precinct"}, errors="raise")
    schools = schools.rename(columns={"Location_Category_Description": "Schools"}, errors="raise")

    schools_g = schools.groupby(['Street Name', 'Issuer Precinct']).count()
    schools_g = schools_g.reset_index()



    df = dd.merge(df, schools_g, how = "left", on=["Street Name", "Issuer Precinct"])
    
    new_name = "with_schools_dask_" + f_name + ".parquet"
    df.to_parquet(new_name)