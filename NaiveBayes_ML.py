##############################
#   Naive Bayes Classifier   #
#              By            #
#    Mohamed Ahmed Khalaf    #
#     Ahmed Galal Ahmed      #
#    Omar Gamal El Gendy     #
#                            #
#         15/12/2016         #
##############################

import scipy
from scipy.signal import iirfilter
import scipy.io as sio
import numpy as np 
from scipy import stats
from scipy.stats import *
from scipy.stats import norm
import matplotlib.pyplot as plt
from IPython.display import *
import math
from math import sqrt , exp , pi
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
%matplotlib inline

#filter things
def notch_filter(data,fs=1000,low=50,high=60, order=7,lowpassFreq=200):
    
    nyq  = 500.0
    low  = 50/nyq
    high = 60/nyq
    
    b, a = iirfilter(order, [low, high], btype='bandstop', analog=False)
    filtered_data = scipy.signal.lfilter(b, a, data)
    
    return filtered_data


palmar = sio.loadmat('C:\Users\M.khalaf\Downloads\Statistics Project_\Statistics Project_\palmar.mat')
p = palmar['palmar']
lateral = sio.loadmat('C:\Users\M.khalaf\Downloads\Statistics Project_\Statistics Project_\lateral.mat')
l= lateral['lateral']

plt.plot(range(3000), p[0,:])
plt.show()

p = notch_filter(p)
l = notch_filter(l)

#plotting after filteration
plt.plot(range(3000), p[0,:])
plt.show()


#using Pandas.DataFrame I'm making Matrices in form of tables
#then I shuffeled and randomzied them with precentage 80% for training

palmar_df = pd.DataFrame(np.array(p))
palmar_df["Class"] = 1
lateral_df = pd.DataFrame(np.array(l))
lateral_df["Class"] = 0
data=palmar_df.append(lateral_df,ignore_index=True)
data = shuffle(data)
msk = np.random.rand(len(data)) < 0.8
train = data[msk]
test = data [~msk]


#reindexing from 0
data.reset_index(inplace=True)
del(data["index"])


#defining the functions for feature selection
def energy(row):
    return sum(row**2)

def fourth_power(row) :
    return sum(row**4)

def non_linear_energy(row):
    s = []
    for index,val in row.iteritems():
        if (index>1)  & (index < (len(row)-1)) :
            s.append((val * row.ix[index-2]*-1) + (row.ix[index-1])**2)
    
    return sum(s)
        
def curve_length(row):
    s = []
    for index,val in row.iteritems():
        if (index>0)  & (index < (len(row)-1)):
             s.append(val - row.ix[index-1])
                
    return sum(s)
    

#Appending the results of functions in new columns in my train table
train["Energy"] = train.apply(energy,axis=1)
train["Fourth Power"] = train.apply(fourth_power,axis=1)
train["Non Linear Energy"] = train.apply(non_linear_energy,axis=1)
train["Curve Length"] = train.apply(curve_length,axis=1)
train.head()


#making train only containing the selected features instead of the 3000 features
train=train[["Energy","Fourth Power","Non Linear Energy","Curve Length","Class"]]
train.head()


#using Pandas to get mean for palmer and putting them in a DataFrame
tmp = train[train["Class"] == 1]
mean =tmp.mean(axis =0)
mean
meanpalmar_df =pd.DataFrame(mean)

meanpalmar_df.drop("Class")


#for mean of Lateral
tmp1 = train[train["Class"] == 0]
mean =tmp1.mean(axis =0)
meanlateral_df =pd.DataFrame(mean)

meanlateral_df.drop("Class")


std=tmp.std(axis=0)
stdpalmar_df=pd.DataFrame(std)
stdpalmar_df.drop("Class")


std=tmp1.std(axis=0)
stdlateral_df=pd.DataFrame(std)
stdlateral_df.drop("Class")


#making the same thing for my test sample
test["Energy"] = test.apply(energy,axis=1)
test["Fourth Power"] = test.apply(fourth_power,axis=1)
test["Non Linear Energy"] = test.apply(non_linear_energy,axis=1)
test["Curve Length"] = test.apply(curve_length,axis=1)
test.head()


#DataFrame of only 4 features and Class
test=test[["Energy","Fourth Power","Non Linear Energy","Curve Length","Class"]]
test.head()


#reindexing from 0
test = test.reset_index()
del (test["index"])

test.head()


#functions to calculate probability of palmar and linear in test sample
P_l=float (test ["Class"][test["Class"]==0] .count())/(test["Class"].count())
P_p=float (test ["Class"][test["Class"]==1] .count())/(test["Class"].count())


#Assuming Normal distribution of selected features
def Calc(index) :
    FP=norm(meanpalmar_df.at['Fourth Power',0],stdpalmar_df.at['Fourth Power',0])
    F_P=FP.pdf(test.at[index,'Fourth Power'])
    EP=norm(meanpalmar_df.at['Energy',0],stdpalmar_df.at['Energy',0])
    E_P=EP.pdf(test.at[index,'Energy'])
    LEP=norm(meanpalmar_df.at['Non Linear Energy',0],stdpalmar_df.at['Non Linear Energy',0])
    LE_P=LEP.pdf(test.at[index,'Non Linear Energy'])
    CP=norm(meanpalmar_df.at['Curve Length',0],stdpalmar_df.at['Curve Length',0])
    C_P=CP.pdf(test.at[index,'Curve Length'])
    FL=norm(meanlateral_df.at['Fourth Power',0],stdlateral_df.at['Fourth Power',0])
    F_L=FL.pdf(test.at[index,'Fourth Power'])
    EL=norm(meanlateral_df.at['Energy',0],stdlateral_df.at['Energy',0])
    E_L=EL.pdf(test.at[index,'Energy'])
    LEL=norm(meanlateral_df.at['Non Linear Energy',0],stdlateral_df.at['Non Linear Energy',0])
    LE_L=LEL.pdf(test.at[index,'Non Linear Energy'])
    CL=norm(meanlateral_df.at['Curve Length',0],stdlateral_df.at['Curve Length',0])
    C_L=CL.pdf(test.at[index,'Curve Length'])
    Prop_of_Palmar=P_p*F_P*E_P*LE_P*C_P
    Prop_of_Lateral=P_l*F_L*E_L*LE_L*C_L
    return Prop_of_Palmar,Prop_of_Lateral


#creating a new column containing the predicted result to compare with test samples
#1 for Palmer and 0 for Lateral 
test["Predicted"] = 0 
for i in test.index:
    Prop_of_Palmar,Prop_of_Lateral = Calc(i)

    if (Prop_of_Lateral < Prop_of_Palmar):
        predicted=1
      
    else :
        predicted=0
       
    test.loc[i,"Predicted"] = predicted
    


#using sklearn to get accuracy
accuracy_score(test.Class,test.Predicted)
