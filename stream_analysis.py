import os
import time

import numpy as np
import pandas as pd
from streamz import Stream
from streamz.dataframe import DataFrame
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import networkx as nx


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
        rmean = ((x * 1) / rmcount) + (rmean * (rmcount - 1)) / rmcount
    return rmean, rmcount, x


# Wikipedia example for online variance, thanks wiki editors
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
# Running avg
# running std dev
# running_
aaa = True
if aaa:

    # source.map(lambda x: get_value(x, "Plate ID")) #za dobit en podatek iz datastreama
    # .accumulate(running_mean, start=[]) #za dobit running mean, count
    # .accumulate(variance_update, start=(0,0,0)).map(get_variance) # za dobit running mean, variance, samplevariance
    # .map(lambda x: get_value(x, "Issue Date")).accumulate(get_tickets_today, start=0) # za dobit current daily tickets čeprav TODO ker ga napake(ne-trenuten datum) sesujejo
    # .accumulate(lambda stt, x: (x if x > stt else stt)) #running max
    # .accumulate(lambda stt, x: (x if x < stt else stt)) #running min
    source.map(lambda x: get_value(x, "Issue Date")).accumulate(get_tickets_today, start=0).sink(print).start()
    business = 0
    for g, chunk in range(business):
        chunk = chunk.to_pandas()
        for i in range(len(chunk)):
            source.emit(chunk.iloc[i])
        print("Batch number", g)
quit()


# TODO stream clustering algo (lahko iz hw4?)


def clust_distance(x, y):
    # x = x[0]
    return np.linalg.norm(x - y)


def clust_distance_vec(x, y):
    out = np.zeros((x.shape[0], y.shape[0]))
    for i1, i in enumerate(x):
        for i2, j in enumerate(y):
            out[i1, i2] = clust_distance(i, j)
    return out


initialized = False

microclusters = []
G = None
nodenames = {}
# TODO EDIT THESE PARAMS FOR NEW DATA
microcluster_size = 3
new_cluster_threshold = 5

# 12/h, 12 * 24/day,
decay = 1 / (12 * 24 * 7)  # 3.5 dni
# G.add_node(1)
removed = []


def calc_loc(loc, ns, x):
    return (loc * ns + x) / (ns + 1)


def update_graph(graph, name):
    # name = name of modified cluster
    distances = np.array([clust_distance(graph.nodes[y]["loc"], graph.nodes[name]["loc"]) for y in graph])
    edges = np.logical_and(0 < distances, distances < (1.5 * microcluster_size))
    if edges.any():
        for x in np.nonzero(edges)[0]:
            if not graph.has_edge(name, x):
                graph.add_edge(name, x, weight=graph.nodes[name]["n_samples"] + graph.nodes[x]["n_samples"] - 1)
            else:
                graph.edges[name, x]["weight"] += 1
        # TODO transform mapping za removane node indekse?


outliers = []
iterations = 0
datapoints = np.zeros((5000, 2))


def CEDAS_cluster(x):
    global initialized
    global G
    global nodenames
    global outliers
    global iterations
    outval = -1
    modified = False
    if not G is None:
        # plt.scatter([G.nodes[node]["loc"][0] for node in G.nodes], [G.nodes[node]["loc"][1] for node in G.nodes])
        # nx.draw(G)
        iterations += 1
        datapoints[iterations % 5000, :] = x
        if iterations % 5000 == 0:
            # for d in datapoints:
            #    distances = np.array([clust_distance(G.nodes[y]["loc"], d) for y in G])

            plt.scatter(datapoints[:, 0], datapoints[:, 1], color="b")
            clustersq = list(nx.connected_components(G))
            for c in enumerate(clustersq):
                plt.scatter([G.nodes[n]["loc"][0] for n in c[1]], [G.nodes[n]["loc"][1] for n in c[1]])
            plt.show()
        # print(len(list(nx.connected_components(G))))
        # plt.show()
    if not initialized:
        G = nx.Graph()
        initialized = True
    if (G is not None and G.number_of_nodes() == 0):

        # G.add_node(0,loc=x, energy=1, n_samples_kern=1, n_samples=1)
        # outliers.append(x)
        # nodenames[0] = x
        # nodenames[x] = 0

        outliers.append(list(x))
        ou = np.array(outliers)
        d = np.atleast_2d(clust_distance_vec(ou, ou))
        NC = np.argmax(np.sum(d < microcluster_size, axis=0))
        if np.sum(d[:, NC] < microcluster_size) > new_cluster_threshold:
            indexes = np.nonzero(d[:, NC] < microcluster_size)

            indexes2 = np.nonzero(d[:, NC] < (microcluster_size / 2))
            outliers = list(np.delete(ou, indexes[0], axis=0))
            G.add_node(len(G), loc=np.mean(ou[indexes[0], :], axis=0), energy=1,
                       n_samples=len(indexes[0]), n_samples_kern=len(indexes2[0]))
            update_graph(G, len(G) - 1)
            modified = True
    else:
        if G is not None and (G.number_of_nodes() > 0):
            distances = np.array([clust_distance(G.nodes[y]["loc"], x) for y in G])
            mind = np.argmin(distances)
            if distances[mind] < microcluster_size:  # within microc
                G.nodes[mind]["energy"] = 1
                G.nodes[mind]["n_samples"] += 1
                if distances[mind] < (microcluster_size / 2):
                    ns = G.nodes[mind]["n_samples_kern"]
                    loc = G.nodes[mind]["loc"]
                    G.nodes[mind]["loc"] = (calc_loc(loc[0], ns, x[0]), calc_loc(loc[1], ns, x[1]))
                    # TODO UPDATE GRAPH
                    update_graph(G, mind)
                    modified = True
                G.nodes[mind]["n_samples"] += 1
            else:  # new microc???
                outliers.append(list(x))
                ou = np.array(outliers)
                d = np.atleast_2d(clust_distance_vec(ou, ou))
                NC = np.argmax(np.sum(d < microcluster_size, axis=0))
                if np.sum(d[:, NC] < microcluster_size) > new_cluster_threshold:
                    indexes = np.nonzero(d[:, NC] < microcluster_size)

                    indexes2 = np.nonzero(d[:, NC] < (microcluster_size / 2))
                    outliers = list(np.delete(ou, indexes[0], axis=0))
                    G.add_node(len(G), loc=np.mean(ou[indexes[0], :], axis=0), energy=1,
                               n_samples=len(indexes[0]), n_samples_kern=len(indexes2[0]))
                    update_graph(G, len(G) - 1)
                    modified = True
        if not G is None:
            distances = np.array([clust_distance(G.nodes[y]["loc"], x) for y in G])
            clusters = list(nx.connected_components(G))
            minind = np.argmin(distances)
            for id, c in enumerate(clusters):
                if minind in c:
                    outval = id

            global decay
            reminds = []
            for node in G.nodes:
                G.nodes[node]["energy"] -= decay
                if G.nodes[node]["energy"] < 0:
                    reminds.append(node)
            if reminds:
                G.remove_nodes_from(reminds)
                newG = nx.Graph()
                cnt = 0
                translation = {}
                for node in G.nodes:
                    translation[node] = cnt
                    newG.add_node(cnt, loc=G.nodes[node]["loc"], energy=G.nodes[node]["energy"],
                                  n_samples=G.nodes[node]["n_samples"], n_samples_kern=G.nodes[node]["n_samples_kern"])
                    cnt += 1
                for edge in G.edges:
                    ed = G.edges[edge]
                    newG.add_edge(translation[edge[0]], translation[edge[1]], weight=ed["weight"])

                G = newG
                print()
            if modified:
                ...

        # now we decay
    return outval


# microclusters.append(("Center", "Count", "MacroCluster"))
# microclusters.append({"center":x, "count":1, "macro_cluster":1, "energy":1, "Edge": 1})
from sklearn.cluster import MiniBatchKMeans

clustering = True
if clustering:
    df_list = pd.read_csv('Parking_Violations_Issued_-_Fiscal_Year_2023.csv', header=0, chunksize=1000)
    # source.map().sink().start()
    cluster = MiniBatchKMeans()
    for chunk in df_list:
        chunk["Issue Date"] = pd.to_datetime(chunk["Issue Date"]).values.astype(int)
        chunk.drop(columns=["Plate ID", "Registration State", "Plate Type", "Vehicle Body Type", "Vehicle Make",
                            "Issuing Agency", "Violation Time", "Street Name", "Intersecting Street",
                            "From Hours In Effect", "To Hours In Effect"])
        cluster.partial_fit(chunk.to_numpy())  # .select_dtypes(include=["float", "int"])

        # for i in range(len(chunk)):
        #    source.emit(chunk["NAMES"].iloc[i])
        # quit()
