# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:11:48 2020

@author: John
"""

from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()
irisDF = pd.DataFrame(iris.data, columns=iris.feature_names)

num_clusters = []
elkan_inertias = []
full_inertias = []

for i in range(1, 6):
    model = KMeans(n_clusters=i, n_init=50, max_iter=500, algorithm="elkan")
    model.fit(irisDF)
    print(i, "->", model.inertia_)
    num_clusters.append(i)
    elkan_inertias.append(model.inertia_)
    model = KMeans(n_clusters=i, n_init=50, max_iter=500, algorithm="full")
    model.fit(irisDF)
    full_inertias.append(model.inertia_)
    
plt.plot(num_clusters, elkan_inertias)
plt.plot(num_clusters, full_inertias)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")    
plt.show()
