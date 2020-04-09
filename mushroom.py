# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 21:50:24 2018

@author: Ryutaro Takanami
"""
from sklearn.decomposition import PCA
import scipy.spatial.distance as sd
import numpy as np
import scipy as sc
import scipy.cluster as scc
from scipy import linalg
import matplotlib.pyplot as plt
import random
import math
import sys
import pandas as pd
from collections import defaultdict



#Load Mashroom
dataRaw = [];
DataFile = open ("agaricus-lepiota.csv", "r")

while True:
    theline = DataFile.readline()
    if len(theline) == 0:
        break
    readData = theline.split(",")
    for pos in range(len(readData)):
        readData[pos] = str(readData[pos]);
    dataRaw.append(readData)
    
DataFile.close()

MashroomData = dataRaw

df = pd.read_csv("agaricus-lepiota.csv")


############################################################################################################################################
#PREPROCESSING


#Check the amount of missing value in case that there are additional one unwritten in description.
print(df.isnull().sum())

#Check the number of value in each value in each column to make sure it is useful for classification
a = df["habitat"].value_counts()
print(a)

#"gill-attachment" actually has only two variables "f" and "a", then the one will be deleted when it become dummy variable
a = df["gill-attachment"].value_counts()
print(a)

#"gill-spacing" actually has only two variables "c" and "w", then the one will be deleted when it become dummy variable
a = df["gill-spacing"].value_counts()
print(a)

#Check the number of value in each value in "stalk-root" for considering about the way to treat missing value
a = df["stalk-root"].value_counts()
print(a)

#"veil-type" has only one value ("partial"), then it will be deleted later
a = df["veil-type"].value_counts()
print(a)



#Check the amount of missing value in the "stalk-root" column
count = 0
for i in range(df.shape[0]):
    if (df.loc[i, "stalk-root"] == "?"):
        count += 1
print(count) #2480

###############################################################################
#(1) delete missing value                                                  
#eliminate the row has missing value
df_dropped = df.copy()
for i in range(df_dropped.shape[0]):
    if (df_dropped.loc[i, "stalk-root"] == "?"):
        df_dropped.drop(index=[i], inplace=True)

#confirm the success of the delete of missing value
#a = df_dropped["stalk-root"].value_counts()
#print(a)

#Convert into dummy variable
df_dropped = pd.get_dummies(df_dropped)
#delete irrelevant columns
df_dropped = df_dropped.drop(["poisonous_e", "bruises_f", "gill-attachment_f", "gill-spacing_w",
                              "gill-size_n", "stalk-shape_t", "veil-type_p"], axis=1)
df_dropped = df_dropped.drop(["cap-shape_s", "cap-surface_s", "cap-color_y", "odor_p", "gill-color_y",
                              "stalk-root_r", "stalk-surface-above-ring_s", "stalk-surface-below-ring_s",
                              "stalk-color-above-ring_y", "stalk-color-below-ring_y", "veil-color_y",
                              "ring-number_t", "ring-type_p", "spore-print_w", "population_y",
                              "habitat_d"], axis=1)

#(2) change the missing values to most frequet value
df_mode = df.copy()
for i in range(df_mode.shape[0]):
    if (df_mode.loc[i, "stalk-root"] == "?"):
        df_mode.loc[i, "stalk-root"] = "b"

#confirm the success of the change to most frequent value
#a = df_mode["stalk-root"].value_counts()
#print(a)

#Convert into dummy variable      
df_mode = pd.get_dummies(df_mode)
#delete irrelevant columns
df_mode = df_mode.drop(["poisonous_e", "bruises_f", "gill-attachment_f", "gill-spacing_w", "gill-size_n", "stalk-shape_t", "veil-type_p"], axis=1)
df_mode = df_mode.drop(["cap-shape_s", "cap-surface_s", "cap-color_y", "odor_p", "gill-color_y",
                              "stalk-root_r", "stalk-surface-above-ring_s", "stalk-surface-below-ring_s",
                              "stalk-color-above-ring_y", "stalk-color-below-ring_y", "veil-color_y",
                              "ring-number_t", "ring-type_p", "spore-print_w", "population_y",
                              "habitat_d"], axis=1)

#(3) deal missing values as one of the categorical data.
df_newCat = df.copy()
for i in range(df_newCat.shape[0]):
    if (df_newCat.loc[i, "stalk-root"] == "?"):
        #change the name of missing values as "m"
        df_newCat.loc[i, "stalk-root"] = "m"     

#confirm the success of the change to most frequent value
a = df_newCat["stalk-root"].value_counts()
print(a)

#Convert into dummy variable
df_newCat = pd.get_dummies(df_newCat)
#delete irrelevant columns
df_newCat = df_newCat.drop(["poisonous_e", "bruises_f", "gill-attachment_f", "gill-spacing_w", "gill-size_n", "stalk-shape_t", "veil-type_p"], axis=1)
df_newCat = df_newCat.drop(["cap-shape_s", "cap-surface_s", "cap-color_y", "odor_p", "gill-color_y",
                              "stalk-root_r", "stalk-surface-above-ring_s", "stalk-surface-below-ring_s",
                              "stalk-color-above-ring_y", "stalk-color-below-ring_y", "veil-color_y",
                              "ring-number_t", "ring-type_p", "spore-print_w", "population_y",
                              "habitat_d"], axis=1)



##################################################################################################

#generate data to implement 5 fold cross validation
def Kfold(df):
    df_randomOrder = df.sample(n=df.shape[0])
    
    
    
    #Divide test and train data
    n = int(df_randomOrder.shape[0]/5)
    test0 = df_randomOrder.iloc[0:n]
    test1 = df_randomOrder.iloc[n:2*n]
    test2 = df_randomOrder.iloc[2*n:3*n]
    test3 = df_randomOrder.iloc[3*n:4*n]
    test4 = df_randomOrder.iloc[4*n:]
    
    #Reset index for later part (for identifying the number of row)
    test0 = test0.reset_index(drop=True)
    test1 = test1.reset_index(drop=True)
    test2 = test2.reset_index(drop=True)
    test3 = test3.reset_index(drop=True)
    test4 = test4.reset_index(drop=True)
    
    train0 = pd.concat([test1, test2, test3, test4], ignore_index = True)
    train1 = pd.concat([test0, test2, test3, test4], ignore_index = True)
    train2 = pd.concat([test0, test1, test3, test4], ignore_index = True)
    train3 = pd.concat([test0, test1, test2, test4], ignore_index = True)
    train4 = pd.concat([test0, test1, test2, test3], ignore_index = True)

    return test0, test1, test2, test3, test4, train0, train1, train2, train3, train4
##################################################################################################################
#KNN


from sklearn.neighbors import KNeighborsClassifier



kNum = list(range(1, 50, 1))
accuracyRate_list = []

for k in kNum:
    #set the handled data as an argument (df_dropped, df_mode, df_newCat)
    test0, test1, test2, test3, test4, train0, train1, train2, train3, train4 = Kfold(df_newCat)
    accuracyRate = 0
    
    
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(train0.drop("poisonous_p", axis=1), train0.poisonous_p)
    pred0 = knc.predict(test0.drop("poisonous_p", axis=1))
    accurate = 0
    for n in range(test0.shape[0]):
        if(test0.poisonous_p[n] == pred0[n]):
            accurate += 1
    accuracyRate += (accurate/test0.shape[0])
    
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(train1.drop("poisonous_p", axis=1), train1.poisonous_p)
    pred1 = knc.predict(test1.drop("poisonous_p", axis=1))
    accurate = 0
    for n in range(test1.shape[0]):
        if(test1.poisonous_p[n] == pred1[n]):
            accurate += 1
    accuracyRate += (accurate/test1.shape[0])
    
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(train2.drop("poisonous_p", axis=1), train2.poisonous_p)
    pred2 = knc.predict(test2.drop("poisonous_p", axis=1))
    accurate = 0
    for n in range(test2.shape[0]):
        if(test2.poisonous_p[n] == pred2[n]):
            accurate += 1
    accuracyRate += (accurate/test2.shape[0])
    
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(train3.drop("poisonous_p", axis=1), train3.poisonous_p)
    pred3 = knc.predict(test3.drop("poisonous_p", axis=1))
    accurate = 0
    for n in range(test3.shape[0]):
        if(test3.poisonous_p[n] == pred3[n]):
            accurate += 1
    accuracyRate += (accurate/test3.shape[0])
    
    
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(train4.drop("poisonous_p", axis=1), train4.poisonous_p)
    pred4 = knc.predict(test4.drop("poisonous_p", axis=1))
    accurate = 0
    for n in range(test4.shape[0]):
        if(test4.poisonous_p[n] == pred4[n]):
            accurate += 1
    accuracyRate += (accurate/test4.shape[0])
    #append the accuracy rate by each the number of neighbors
    accuracyRate_list.append(accuracyRate/5)


plt.plot(kNum, accuracyRate_list)
plt.xlabel("The number of nearest neighbors")
plt.ylabel("The accuracy rate")


"""
knc = KNeighborsClassifier(n_neighbors=5)
knc.fit(train0.drop("poisonous_p", axis=1), train0.poisonous_p)
pred = knc.predict(test0.drop("poisonous_p", axis=1))

#Check the accuracy
accurate = 0
count = 0
for n in range(test0.shape[0]):
    if(test0.poisonous_p[n] == pred[n]):
        accurate += 1

accuracyRateKNN = accurate/test0.shape[0]
"""


######################################################################################################################
#SVM



from sklearn.svm import SVC

#set the handled data as an argument (df_dropped, df_mode, df_newCat)
test0, test1, test2, test3, test4, train0, train1, train2, train3, train4 = Kfold(df_newCat)
accuracyRate = 0
    
    
svm = SVC(kernel='linear', random_state=None)
svm.fit(train0.drop("poisonous_p", axis=1), train0.poisonous_p)
pred0 = svm.predict(test0.drop("poisonous_p", axis=1))
accurate = 0
for n in range(test0.shape[0]):
    if(test0.poisonous_p[n] == pred0[n]):
        accurate += 1
accuracyRate += (accurate/test0.shape[0])

svm = SVC(kernel='linear', random_state=None)
svm.fit(train1.drop("poisonous_p", axis=1), train1.poisonous_p)
pred1 = svm.predict(test1.drop("poisonous_p", axis=1))
accurate = 0
for n in range(test1.shape[0]):
    if(test1.poisonous_p[n] == pred1[n]):
        accurate += 1
accuracyRate += (accurate/test1.shape[0])

svm.fit(train2.drop("poisonous_p", axis=1), train2.poisonous_p)
pred2 = svm.predict(test2.drop("poisonous_p", axis=1))
accurate = 0
for n in range(test2.shape[0]):
    if(test2.poisonous_p[n] == pred2[n]):
        accurate += 1
accuracyRate += (accurate/test2.shape[0])

svm.fit(train3.drop("poisonous_p", axis=1), train3.poisonous_p)
pred3 = svm.predict(test3.drop("poisonous_p", axis=1))
accurate = 0
for n in range(test3.shape[0]):
    if(test3.poisonous_p[n] == pred3[n]):
        accurate += 1
accuracyRate += (accurate/test3.shape[0])

svm.fit(train4.drop("poisonous_p", axis=1), train4.poisonous_p)
pred4 = svm.predict(test4.drop("poisonous_p", axis=1))
accurate = 0
for n in range(test4.shape[0]):
    if(test4.poisonous_p[n] == pred4[n]):
        accurate += 1
accuracyRate += (accurate/test4.shape[0])

#append the accuracy rate by each fold
accuracyRateCV = accuracyRate/5
#print the cross validated accuracy rate
print(accuracyRateCV)



"""
svm = SVC(kernel='linear', random_state=None)
svm.fit(train0.drop("poisonous_p", axis=1), train1.poisonous_p)
pred = svm.predict(test1.drop("poisonous_p", axis=1))

#Check the accuracy
accurate = 0
for n in range(test1.shape[0]):
    if(test1.poisonous_p[n] == pred[n]):
        accurate += 1

accuracyRateSVM = accurate/test1.shape[0]
"""

##################################################################################################################################################
#RBFNN(http://www.brain.kyutech.ac.jp/~furukawa/data/rbf.html)


"""
def dist(p1,p2):
    sumTotal = 0
    
    for c in range(len(p1)):
        sumTotal = sumTotal + pow((p1[c] - p2[c]),2)
    
    return math.sqrt(sumTotal)

def maxDist(m1,m2):
    maxDist = -1
    
    for i in range(len(m1)):
        for j in range(len(m2)):
            distance = dist(m1[i,:],m2[j,:])
            
            if(distance > maxDist):
                maxDist = distance
    
    return maxDist




def RBFTrain(data, labels):
    ## Converting labels
    convLabels = []
    
    for label in labels:
        if(label == 0):
            convLabels.append([1, 0, 0])
        elif (label == 1):
            convLabels.append([0, 1, 0])
        else:
            convLabels.append([0, 0, 1])

    group1 = np.random.randint(0, 45, size=nPrototypes)
    group2 = np.random.randint(45, 90, size=nPrototypes)
    group3 = np.random.randint(90, 135, size=nPrototypes)
    
    prototypes = np.vstack([data[group1, :], data[group2, :], data[group3, :]])
    
    distance = maxDist(prototypes, prototypes)
    sigma = distance / math.sqrt(nPrototypes*nClasses)
    
    dataRows = data.shape[0]
    
    output = np.zeros(shape=(dataRows, nPrototypes * nClasses))
    
    for item in range(dataRows):
        out = []
        
        for proto in prototypes:
            distance = dist(data[item], proto)
            neuronOut = np.exp(-distance / np.square(sigma))
            out.append(neuronOut)
            
        output[item, :] = np.array(out)
    weights = np.dot(pinv(output), convLabels)
    
    return weights, prototypes, sigma



def RBFPredict(item, prototype, weights, sigma):
    out = []
    
    ## Hidden layer
    for proto in prototypes:
        distance = dist(item, proto)
        neuronOut = np.exp(-(distance) / np.square(sigma))
        out.append(neuronOut)
    
    netOut = []
    for c in range(nClasses):
        result = np.dot(weights[:,c],out)
        netOut.append(result)
        
    return np.argmax(netOut)

weights, prototypes, sigma = RBFTrain(trainingData, trainingLabels)

for item in testItems:
    predictClass = RBFPredict(data[item,:], prototypes, weights, sigma)
    print("Item: " + str(item))
    print("Predicted Class: " + str(predictClass))
    print("True Class: " + str(labels[item]))


nPrototypes = 4
nClasses = 3


"""













"""
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
"""
#print(eliminatedData[18,:])

"""
# test eliminate_outlier() for one column
sigma = np.std(pulsarData[:, 1])
mu = np.mean(pulsarData[:, 1])
eliminatedData = pulsarData[(abs(pulsarData[:, 1] - mu) / sigma) <= 3]
"""

"""
count = 0
for i in range(eliminatedData.shape[0]):
    if(eliminatedData[i, 8] == 1):
        count += 1
print(count)
eliminatedData = eliminatedData
# The number of pulsar in outlier of each variable
# 1:447, 2:15, 3:631, 4:523, 5:271, 6:82, 7:1, 8:4, ALL: 772(original:1639)
"""


"""
plt.hist(pulsarData[:, 1], bins = 30)
plt.hist(eliminatedData[:, 1], bins = 20)
"""





"""

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
normalisedData = normalise(eliminatedData)
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
#distanceData = distance(data)

# need to expand limitation for recursive
sys.setrecursionlimit(15000)
condensedDistance = sd.squareform(distanceData)
Z = scc.hierarchy.linkage(condensedDistance, method="ward")

result = scc.hierarchy.fcluster(Z, t=2, criterion="maxclust")

count = 0

d = defaultdict(list)
for i, r in enumerate(result):
    d[r].append(i)
for k, v in d.items():
    print(k, v)
    for i in v:
        if(normalisedData[i, 8] == 1):
            count += 1
    print(count)




plt.figure(figsize=(6,4))
#plt.figure(figsize=(300, 100))
scc.hierarchy.dendrogram(Z, truncate_mode="level", p = 50)



"""






































