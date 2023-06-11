import pandas as pd


# with pd.read_csv('/d/hpc/projects/FRI/bigdata/data/NYTickets/2023_april.csv', dtype=str, header=0, chunksize=100000) as reader:
#     for i, chunk in enumerate(reader):
#         chunk.to_hdf("./data/parking.hdf5", key="chunk_" + str(i), mode="a", format='table')



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
cols = ['Street Name', 'Issuer Precinct', 'Issue Date', 'Violation Time', "Summons Number"]
# path = "/d/hpc/projects/FRI/bigdata/data/NYTickets/2023_april.csv"
path = "./data/parking.hdf5"
df = dd.read_hdf(path, key="*", columns=cols,
)

df = df.repartition(npartitions=16)
# df2 = df2.repartition(npartitions=1)

df['Issuer Precinct'] = df['Issuer Precinct'].astype(float)

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

schools = pd.read_csv('./data/school_locations.csv')
def street_from_address(x):
    # make sure the addresses are written correctly
    a = x.split(" ")[1:]
    return ' '.join(a)

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

df.to_hdf("with_schools_hdf.hdf5", key="data")