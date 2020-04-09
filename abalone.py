# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 21:50:24 2018

@author: Ryutaro Takanami


@relation abalone19
@attribute Sex {M, F, I}
@attribute Length real [0.075, 0.815]
@attribute Diameter real [0.055, 0.65]
@attribute Height real [0.0, 1.13]
@attribute Whole_weight real [0.0020, 2.8255]
@attribute Shucked_weight real [0.0010, 1.488]
@attribute Viscera_weight real [5.0E-4, 0.76]
@attribute Shell_weight real [0.0015, 1.005]
@attribute Class {positive, negative}
@inputs Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight
@outputs Class
@data

"""
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd



#Load Abalone

df = pd.read_csv("abalone19.csv")

#one-hot encoding to Sex(the only categorical data in this dataset)
df = pd.get_dummies(df, columns=["Sex"])

#Check the amount of missing value in case that there are additional one unwritten in description.
#print(df.isnull().sum())

#Counts the number of each value in "Class" and make sure the degree of imbalance
print(df['Class'].value_counts())



#Over Sampling
def overSampling(data, k):
    #make dataframe which class is only positive
    minorityData = data.query('Class == " positive"')
    #make dataframe which expand the size of positie data randomly
    addData = minorityData.sample(n=(k-minorityData.shape[0]), replace=True)
       
    #conbine original data and oversampled data
    newData = pd.concat([data, addData], ignore_index = True)

    return newData
    




#Under Sampling
#the argument of "rate" in percentage of the negative class against the positive class
def UnderSampling(data, rate):
    #make dataframe which is only positive/negative data
    minorityData = data.query('Class == " positive"')
    majorityData = data.query('Class == "negative"')
    #make dataframe which reduce the size of positie data randomly
    underSampledData = majorityData.sample(n=int(minorityData.shape[0]*(rate/100)), replace=True)
    
    #conbine minority and undersampled data
    newData = pd.concat([minorityData, underSampledData], ignore_index = True)
    
    return newData



####################################################################################
#create test and train data

#devide positive class into 4 data
test_p = df.query('Class == " positive"')
test0_p = test_p[0:8]
test1_p = test_p[8:16]
test2_p = test_p[16:24]
test3_p = test_p[24:32]

#create positive train data by comvining positive test data
train0_p = pd.concat([test1_p, test2_p, test3_p], ignore_index = True)
train1_p = pd.concat([test0_p, test2_p, test3_p], ignore_index = True)
train2_p = pd.concat([test0_p, test1_p, test3_p], ignore_index = True)
train3_p = pd.concat([test0_p, test1_p, test2_p], ignore_index = True)

#oversampling
positiveNum = 100
test0_p = overSampling(test0_p, positiveNum)
test1_p = overSampling(test1_p, positiveNum)
test2_p = overSampling(test2_p, positiveNum)
test3_p = overSampling(test3_p, positiveNum)

train0_p = overSampling(train0_p, test0_p.shape[0]*3)
train1_p = overSampling(train1_p, test1_p.shape[0]*3)
train2_p = overSampling(train2_p, test2_p.shape[0]*3)
train3_p = overSampling(train3_p, test3_p.shape[0]*3)

#devide negative class into 4 data
test_n = df.query('Class == "negative"')
test0_n = test_n[0:1035]
test1_n = test_n[1035:2070]
test2_n = test_n[2070:3105]
test3_n = test_n[3105:]

#create negative train data by comvining negative test data
train0_n = pd.concat([test1_n, test2_n, test3_n], ignore_index = True)
train1_n = pd.concat([test0_n, test2_n, test3_n], ignore_index = True)
train2_n = pd.concat([test0_n, test1_n, test3_n], ignore_index = True)
train3_n = pd.concat([test0_n, test1_n, test2_n], ignore_index = True)

#combine positive and negative test data
test0 = pd.concat([test0_p, test0_n], ignore_index = True)
test1 = pd.concat([test1_p, test1_n], ignore_index = True)
test2 = pd.concat([test2_p, test2_n], ignore_index = True)
test3 = pd.concat([test3_p, test3_n], ignore_index = True)

#combine positive and negative train data
train0 = pd.concat([train0_p, train0_n], ignore_index = True)
train1 = pd.concat([train1_p, train1_n], ignore_index = True)
train2 = pd.concat([train2_p, train2_n], ignore_index = True)
train3 = pd.concat([train3_p, train3_n], ignore_index = True)

#########################################################################################################
#KNN


from sklearn.neighbors import KNeighborsClassifier




#optimize the parameter of k in the KNN
negativeRate = 2000
#the range of the k
kNum = list(range(1, 200, 1))
accuracyRate_list = []

for k in kNum:
    #undersampling
    underTest0 = UnderSampling(test0, negativeRate)
    underTest1 = UnderSampling(test1, negativeRate)
    underTest2 = UnderSampling(test2, negativeRate)
    underTest3 = UnderSampling(test3, negativeRate)
    
    underTrain0 = UnderSampling(train0, negativeRate)
    underTrain1 = UnderSampling(train1, negativeRate)
    underTrain2 = UnderSampling(train2, negativeRate)
    underTrain3 = UnderSampling(train3, negativeRate)
    
    accuracyRate = 0
    
    #train the data, predict the label and calcurate the accuracy rate by each fold
    #define the number of nearest neighbours
    knc = KNeighborsClassifier(n_neighbors=k)
    #train the data
    knc.fit(underTrain0.drop("Class", axis=1), underTrain0.Class)
    #predict the lavel
    pred0 = knc.predict(underTest0.drop("Class", axis=1))
    #calcurate the accuracy rate
    accurate = 0
    for n in range(underTest0.shape[0]):
        if(underTest0.Class[n] == pred0[n]):
            accurate += 1
    accuracyRate += (accurate/underTest0.shape[0])
    
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(underTrain1.drop("Class", axis=1), underTrain1.Class)
    pred1 = knc.predict(underTest1.drop("Class", axis=1))
    accurate = 0
    for n in range(underTest1.shape[0]):
        if(underTest1.Class[n] == pred1[n]):
            accurate += 1
    accuracyRate += (accurate/underTest1.shape[0])
    
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(underTrain2.drop("Class", axis=1), underTrain2.Class)
    pred2 = knc.predict(underTest2.drop("Class", axis=1))
    accurate = 0
    for n in range(underTest2.shape[0]):
        if(underTest2.Class[n] == pred2[n]):
            accurate += 1
    accuracyRate += (accurate/underTest2.shape[0])
    
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(underTrain3.drop("Class", axis=1), underTrain3.Class)
    pred3 = knc.predict(underTest3.drop("Class", axis=1))
    accurate = 0
    for n in range(underTest3.shape[0]):
        if(underTest3.Class[n] == pred3[n]):
            accurate += 1
    accuracyRate += (accurate/underTest3.shape[0])
    
    #append the accuracy rate by each the number of neighbors
    accuracyRate_list.append(accuracyRate/4)


plt.plot(kNum, accuracyRate_list)
plt.xlabel("The number of nearest neighbors")
plt.ylabel("The accuracy rate")




#test the percentage of the negative class
knc = KNeighborsClassifier(n_neighbors=175)

x = list(range(10, 5000, 10))
accuracyRate_list = []


for negativeRate in x:
    #undersampling
    underTest0 = UnderSampling(test0, negativeRate)
    underTest1 = UnderSampling(test1, negativeRate)
    underTest2 = UnderSampling(test2, negativeRate)
    underTest3 = UnderSampling(test3, negativeRate)
    
    underTrain0 = UnderSampling(train0, negativeRate)
    underTrain1 = UnderSampling(train1, negativeRate)
    underTrain2 = UnderSampling(train2, negativeRate)
    underTrain3 = UnderSampling(train3, negativeRate)
    
    #Cross validation
    #train the data, predict the label and calcurate the accuracy rate by each fold
    accuracyRate = 0
    #train the data
    knc.fit(underTrain3.drop("Class", axis=1), underTrain0.Class)
    #predict the lavel
    pred0 = knc.predict(underTest0.drop("Class", axis=1))
    #calcurate the accuracy rate
    accurate = 0
    for n in range(underTest0.shape[0]):
        if(underTest0.Class[n] == pred0[n]):
            accurate += 1
    accuracyRate += (accurate/underTest0.shape[0])
    
    
    knc.fit(underTrain1.drop("Class", axis=1), underTrain1.Class)
    pred1 = knc.predict(underTest1.drop("Class", axis=1))
    accurate = 0
    for n in range(underTest1.shape[0]):
        if(underTest1.Class[n] == pred1[n]):
            accurate += 1
    accuracyRate += (accurate/underTest1.shape[0])
    
    knc.fit(underTrain2.drop("Class", axis=1), underTrain2.Class)
    pred2 = knc.predict(underTest2.drop("Class", axis=1))
    accurate = 0
    for n in range(underTest2.shape[0]):
        if(underTest2.Class[n] == pred2[n]):
            accurate += 1
    accuracyRate += (accurate/underTest2.shape[0])
    
    knc.fit(underTrain3.drop("Class", axis=1), underTrain3.Class)
    pred3 = knc.predict(underTest3.drop("Class", axis=1))
    accurate = 0
    for n in range(underTest3.shape[0]):
        if(underTest3.Class[n] == pred3[n]):
            accurate += 1
    accuracyRate += (accurate/underTest3.shape[0])
    
    #append the accuracy rate by each negative class rate
    accuracyRate_list.append(accuracyRate/4)

plt.plot(x, accuracyRate_list)
plt.xlabel("The percentage of negative class")
plt.ylabel("The accuracy rate")


#########################################################################################################
#SVM


from sklearn.svm import SVC





svm = SVC(kernel='linear', random_state=None)
#the list of the rate of the number of the negative class
x = list(range(10, 5000, 10))
accuracyRate_list = []


for negativeRate in x:
    underTest0 = UnderSampling(test0, negativeRate)
    underTest1 = UnderSampling(test1, negativeRate)
    underTest2 = UnderSampling(test2, negativeRate)
    underTest3 = UnderSampling(test3, negativeRate)
    
    underTrain0 = UnderSampling(train0, negativeRate)
    underTrain1 = UnderSampling(train1, negativeRate)
    underTrain2 = UnderSampling(train2, negativeRate)
    underTrain3 = UnderSampling(train3, negativeRate)
    
    #Cross validation
    #train the data, predict the label and calcurate the accuracy rate by each fold
    accuracyRate = 0
    #train the data
    svm.fit(underTrain0.drop("Class", axis=1), underTrain0.Class)
    #predict the lavel
    pred0 = svm.predict(underTest0.drop("Class", axis=1))
    #calcurate the accuracy rate
    accurate = 0
    for n in range(underTest0.shape[0]):
        if(underTest0.Class[n] == pred0[n]):
            accurate += 1
    accuracyRate += (accurate/underTest0.shape[0])
    
    
    svm.fit(underTrain1.drop("Class", axis=1), underTrain1.Class)
    pred1 = svm.predict(underTest1.drop("Class", axis=1))
    accurate = 0
    for n in range(underTest1.shape[0]):
        if(underTest1.Class[n] == pred1[n]):
            accurate += 1
    accuracyRate += (accurate/underTest1.shape[0])
    
    svm.fit(underTrain2.drop("Class", axis=1), underTrain2.Class)
    pred2 = svm.predict(underTest2.drop("Class", axis=1))
    accurate = 0
    for n in range(underTest2.shape[0]):
        if(underTest2.Class[n] == pred2[n]):
            accurate += 1
    accuracyRate += (accurate/underTest2.shape[0])
    
    svm.fit(underTrain3.drop("Class", axis=1), underTrain3.Class)
    pred3 = svm.predict(underTest3.drop("Class", axis=1))
    accurate = 0
    for n in range(underTest3.shape[0]):
        if(underTest3.Class[n] == pred3[n]):
            accurate += 1
    accuracyRate += (accurate/underTest3.shape[0])
    
    #append the accuracy rate by each negative class rate
    accuracyRate_list.append(accuracyRate/4)

plt.plot(x, accuracyRate_list)
plt.xlabel("The percentage of negative class")
plt.ylabel("The accuracy rate")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    