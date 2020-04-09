# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 21:50:24 2018

@author: Ryutaro Takanami
"""
import scipy.spatial.distance as sd
import numpy as np
import scipy as sc
import scipy.cluster as scc
from scipy import linalg
import matplotlib.pyplot as plt
import random
import math
import sys
from collections import defaultdict

#Load Pulsar
dataRaw = [];
DataFile = open ("HTRU_2.csv", "r")

while True:
    theline = DataFile.readline()
    if len(theline) == 0:
        break
    readData = theline.split(",")
    for pos in range(len(readData)):
        readData[pos] = float(readData[pos]);
    dataRaw.append(readData)
    
DataFile.close()

pulsarData = np.array(dataRaw)


def eliminate_outlier(data):
    eliminatedData = data.copy()
    
    cols = data.shape[1]-1
    print(cols)
    
    for j in range(cols):
        sigma = np.std(data[:, j])
        mu = np.mean(data[:, j])
        
        eliminatedData = eliminatedData[(abs(eliminatedData[:, j] - mu) / sigma) <= 3]
    
    return eliminatedData

eliminatedData = eliminate_outlier(pulsarData)


count = 0
for i in range(eliminatedData.shape[0]):
    if(eliminatedData[i, 8] == 1):
        count += 1
print(count)

# The number of pulsar in outlier of each variable
# 1:447, 2:15, 3:631, 4:523, 5:271, 6:82, 7:1, 8:4, ALL: 772(original:1639)

#Make test data with eliminating label
test_data = eliminatedData[:, 0:8]
label = eliminatedData[:, 8]

"""
plt.hist(pulsarData[:, 1], bins = 30)
plt.hist(eliminatedData[:, 1], bins = 20)
"""




###########################################################################################

#Hierarchical clustering

def normalise(data):
    normalisedData = data.copy()
    rows = data.shape[0]
    cols = data.shape[1]-1
    
    for j in range(cols):
        maxElement = np.amax(data[:,j])
        minElement = np.amin(data[:,j])
        
        for i in range(rows):
            normalisedData[i,j] = (data[i,j] - minElement) / (maxElement - minElement)
    
    return normalisedData


# normalize before Clustering
normalisedData = normalise(test_data)
#plt.hist(normalisedData[:, 1], bins = 20)

# Clustering
def distance(data):
    rows = data.shape[0]
    cols = data.shape[1]
    
    distanceMatrix = np.zeros((rows, rows))
    
    for i in range(rows):
        print(i)
        for j in range(rows):
            
            sumTotal = 0
            
            for c in range(cols):
                
                sumTotal = sumTotal + pow((data[i, c] - data[j, c]), 2)
            
            distanceMatrix[i,j] = math.sqrt(sumTotal)
    
    return distanceMatrix


data = np.array(normalisedData)
distanceData = distance(data)


# need to expand limitation for recursive
sys.setrecursionlimit(15000)
condensedDistance = sd.squareform(distanceData)

#"linkage" function takes the condensed distatnce information and links pairs of observations and clusters.
#for euclidean distance
#Z = scc.hierarchy.linkage(condensedDistance)

#for ward method
Z = scc.hierarchy.linkage(condensedDistance, method="ward")

pred_label = scc.hierarchy.fcluster(Z, t=2, criterion="maxclust")
#convert 2 to 0 because that is suit to compare with label
pred_label = np.where(pred_label == 2, 0, 1)


#compute the accuracy rate
accurate = 0
for n in range(len(label)):
    if label[n] == pred_label[n]:
        accurate += 1
accuracyRate = accurate/len(label)
print(accuracyRate)



#plot dendrogram
plt.figure(figsize=(6,4))
scc.hierarchy.dendrogram(Z, truncate_mode="level", p = 10)


#####################################################################################
#K-means Clustering
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA

def standard(data):
    standardData = data.copy()
    
    # the number of rows
    rows = data.shape[0]
    #the number of columns
    cols = data.shape[1]
    
    for j in range(cols):
        #computes the standard deviation
        sigma = np.std(data[:,j])
        #computes the mean
        mu = np.mean(data[:,j])
        
        for i in range(rows):
            #standardising
            standardData[i,j] = (data[i,j] - mu)/sigma
        
    return standardData

#standardizing before the clustering
StandardisedData = standard(test_data)

#set k (the number of clusters) as 2 to divide pulsars and not pulsars
kmeans_model = KMeans(n_clusters=2, random_state=10).fit(StandardisedData)


#obtain predicted labels
pred_label = kmeans_model.labels_
#Seek Centroids
centroids = kmeans_model.cluster_centers_

accurate = 0
for n in range(len(label)):
    if label[n] == pred_label[n]:
        accurate += 1
accuracyRate = accurate/len(label)
print(accuracyRate)
   


#Use PCA to make easy to see the clustered data
pca2 = PCA(n_components=2)
pca2.fit(StandardisedData)
#transform to 2 dimentions
Stand_trans = pca2.fit_transform(StandardisedData)
StaLabelData = np.insert(Stand_trans,2, pred_label, axis=1)

#use PCA to centroids
pca_cent = PCA(n_components=2)
pca_cent.fit(centroids)
#transform to 2 dimentions
centroids_trans = pca2.fit_transform(centroids)

#divide data by label to plot by different color
cluster0 = (StaLabelData[StaLabelData[:,2]==0, :])
plt.scatter(cluster0[:,0],cluster0[:,1], c="red")

cluster1 = (StaLabelData[StaLabelData[:,2]==1, :])
plt.scatter(cluster1[:,0],cluster1[:,1], c="blue")

#plot centroids as x-marker
plt.plot(centroids_trans[0,0], centroids_trans[0, 1], 'yx')
plt.plot(centroids_trans[1,0], centroids_trans[1, 1], 'gx')
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")



























