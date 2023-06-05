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


#client = Client(cluster, timeout="120s")
cols = ['Street Name', 'Issuer Precinct', 'Issue Date', 'Violation Time', 'Violation County']
path = "/d/hpc/projects/FRI/bigdata/data/NYTickets/2023_april.csv"
df = dd.read_csv(path, assume_missing = True, usecols=cols,
dtype={
    'Issuer Precinct': np.float64,

})

df = df.repartition(npartitions=16)
# df2 = df2.repartition(npartitions=1)

print("read data")

df['Street Name'] = df['Street Name'].astype(str)

f = open('data/street-suffix-abbreviations.json')
street_suffix = json.load(f)

def fix_street_abbreviation(x):
    abb = str(x).split()[-1]

    if abb in street_suffix:
        x = x.replace(abb, street_suffix[abb][0])
    return x

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

#df["Violation County"] = df["Violation County"].astype('str')
df["Address Borough"] = df.apply(lambda row: county_to_borough(str(row['Violation County'])), axis = 1, meta=('address', 'str'))

business = business[["Street Name", "Address Borough", "Business"]]

business_g = business.groupby(['Street Name', 'Address Borough']).count()
business_g = business_g.reset_index()

df = dd.merge(df, business_g, how = "left", on=["Street Name", "Address Borough"])

df['Business'] = df['Business'].fillna(0)

df.to_parquet("with_business_dask.parquet")