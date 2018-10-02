
#import the essential libraries for data preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset

dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


#handling missing data in the dataset
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
#replacing the missing values with the mean of the column attribute values
#different strategy most frequent and median
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])


#encoding catagorical data in the dataset
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])

#to avoid hierarchial difference
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])
#index of the feature to be encoded this is to be done after label encoding
X=onehotencoder.fit_transform(X).toarray()

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

#splitting the dataset into the test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

#scaling
'''Xstand=X-mean(X)/standard_deviation(X)
normalisation
Xnorm=x-min(X)/max(X)-min(X)
'''
#feature scaling

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

