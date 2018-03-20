#Data Preprocessing 

#Importing the libraries

#three most essential libraries 

import numpy as np #np is a shortcut
#contains mathematical tools 
import matplotlib.pyplot as plt #plt is shourtcut name
#pyplot is a sub library 
#this library helps to plot charts 
import pandas as pd #pd is shortcut name
#to import and manage data sets 

#importing the dataset 
dataset = pd.read_csv('Data.csv')
#using pandas to import data set in the folder 
X = dataset.iloc[:, : -1].values 
#Create a matrix for all independent variables 
#and put that matrix in x
#:, : take all the lines, columns 
#-1 means donot take the last column
#.values is take all the values 
#first index starts at 0
Y = dataset.iloc[:, 3 ].values
#independent variable matrix 
#the independent variable is in 3rd column 
 

#taking care of missing data 
#for this we take the average of available data and imput 
#that to the missing block 
from sklearn.preprocessing import Imputer 
#sikat learn  has great tools for machine learning 
#preprocessing is the sublibrary 
#imputer is used to handel the problem of missing data 
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
#NaN because the missing values are represented by nan 
#statery is to take the mean (default)
#axis=0, then mean of columns is used (default)
#axis=1, then mean of rows is used 
#imputer = Imputer(missing_values = 'NaN')
#this code is same as above
imputer= imputer.fit(X[:, 1:3])
#fit the imputer calculatevalue in all lines 
#in column 1 and 2 
#1:3 start from 1 until 2
X[:, 1:3] = imputer.transform(X[:, 1:3])
#to replace the matrix in X with the mean value 

#categorical variables 
#country and purchased are two categorical variables 
#country has the name fo different countries(categories)
#purchased has 2 categories (yes and no)
#we do this because we need numbers for machine learning
#and not text 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_country = LabelEncoder()
#create a object
labelencoder_country.fit_transform(X[:, 0])
#this gives the encoded values of the countries
X[:,0] = labelencoder_country.fit_transform(X[:,0])
#putting encoded values for countries in X 
#this creates a problem as every coutry gets a specific value 
#and sets a order 
#to solve thsi we require dummyvariable 
#that is each category has a seperate column 

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#every column correspond to a specific category 

#for purchased 
labelencoder_purchased = LabelEncoder()
labelencoder_purchased.fit_transform(Y)
#this gives the encoded values of the purchases
Y = labelencoder_purchased.fit_transform(Y)
#purchased has only one column 
#only two cagetories so we donot need dummy variables 

#splitting dataset to training set and test set

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, 
                                                    )
#testsize means of the total values in the dataset 
#How many values is used for testing 
#0.2 = 20 %
#mostly tetsting value is less than 50% at max 40 %
#and common test value is about 10 and 20 %
#this gives us 4 different variables for test and train 

#feature scaling 
#machine learing is based on euclidean distance 
#that is distance between two data points
#the value of the salary ranges very high compared to age
#this will cause salary to dominate the value of age
#it would mean that age doesnot even exist 
#we can do this by normalization or standarization 
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
#creating a object
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#for test set we donot need to fit as it is already 
#fitted on the training set 

