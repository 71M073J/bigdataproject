import pandas as pd
import json


import numpy as np
from dask import dataframe as dd
from distributed import Client
from dask_jobqueue import SLURMCluster


def county_to_borough(x):
    if (x == 'K') | (x.upper() == 'KINGS') | (x == 'BK'):
        return 'BROOKLYN'
    elif (x == 'BX') | (x.upper() == 'BRONX'):
        return 'BRONX'
    elif (x == 'R') | (x.upper() == 'RICH') | (x == 'ST') | (x.upper() == 'RICHM'):
        return 'STATEN ISLAND'
    elif (x == 'NY') | (x == 'MN'):
        return 'MANHATTAN'
    elif (x == 'Q') | (x == 'QN') | (x.upper() == 'QNS'):
        return 'QUEENS'
    else:
        return x 
        
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

cols = ['Street Name', 'Issuer Precinct', 'Issue Date', 'Violation Time', 'Violation County', "Summons# Number"]

path = "/d/hpc/projects/FRI/bigdata/data/NYTickets/"

#file_name = "2023_april"


names = ["2023_april", "2022_full", "2021", "2020", "2019", "2018"]

for f_name in names:
    
    full_path = path+f_name+".csv"
    df = dd.read_csv(full_path, assume_missing = True, usecols=cols, dtype={
    'Issuer Precinct': np.float64, })
  
    df = df.repartition(npartitions=16)
# df2 = df2.repartition(npartitions=1)

    print("read data")

    df['Street Name'] = df['Street Name'].astype(str)

    f = open('data/street-suffix-abbreviations.json')
    street_suffix = json.load(f)


    df["Street Name"] = df.apply(lambda row: fix_street_abbreviation(row['Street Name']), axis = 1, meta=('Street Name', 'str'))


    b_cols = ['Address Borough', 'Address Street Name', 'License Type', 'License Status', 'Address State', ]
    business = pd.read_csv('./data/legally_operating_businesses.csv', usecols=b_cols)

    # filter data - active businesses only, ONLY in new york
    business = business[(business["License Type"] == "Business") & (business["License Status"] == "Active") & (business["Address State"] == "NY") & (business["Address Borough"] != "Outside NYC")]

    business["Address Borough"] = business["Address Borough"].str.upper()

    business["Address Street Name"] = business["Address Street Name"].str.upper()
    business["Address Street Name"] = business["Address Street Name"].astype('str')

    business["Address Street Name"] = business.apply(lambda row: fix_street_abbreviation(row['Address Street Name']), axis = 1)

    business = business.rename(columns={"Address Street Name": "Street Name"})

    business['Business'] = 1

       

    #df["Violation County"] = df["Violation County"].astype('str')
    df["Address Borough"] = df.apply(lambda row: county_to_borough(str(row['Violation County'])), axis = 1, meta=('address', 'str'))

    business = business[["Street Name", "Address Borough", "Business"]]

    business_g = business.groupby(['Street Name', 'Address Borough']).count()
    business_g = business_g.reset_index()

    df = dd.merge(df, business_g, how = "left", on=["Street Name", "Address Borough"])

    df['Business'] = df['Business'].fillna(0)

    new_name = "with_business_dask_" + f_name + ".parquet"
    df.to_parquet(new_name)
      
