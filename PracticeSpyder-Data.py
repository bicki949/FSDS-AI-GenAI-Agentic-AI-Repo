import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset=pd.read_csv(r"D:\Download\ML-DATA\Data.csv")

x = dataset.iloc[:,:-1].values #indipendent variable

y = dataset.iloc[:,3].values #dipendent variable

# SKLEARN FILL MISSING NUMERICAL VALUE

from sklearn.impute import SimpleImputer

# sklearn.impute-- transformer to handle fill missing value
# SimpleImputer(Library)-- This library fill missing value
# parameter tunning apply-mean startegy
# hyperparameter-tunning-median/mode /most_frequent startegy

imputer = SimpleImputer(strategy='most_frequent')

imputer = imputer.fit(x[:,1:3])

x[:,1:3] = imputer.transform(x[:,1:3])




#NEW LIBRARY
from sklearn.preprocessing import LabelEncoder
#LabelEncoder=it transformer form to fill categorical to numerical data
labelEncoder_x = LabelEncoder()

x[:,0]=labelEncoder_x.fit_transform(x[:,0])


labelEncoder_y=LabelEncoder()

y=labelEncoder_y.fit_transform(y)

#SPLIT THE DATA

from sklearn.model_selection import  train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y,train_size=0.7,test_size=0.3,random_state=0)















