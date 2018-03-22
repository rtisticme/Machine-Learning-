#multiple Linear REgression 
#To see the profit generated with different investments 

# Data Preprocessing 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#encoding the categorical variable 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_state = LabelEncoder()
X[:,3]=labelencoder_state.fit_transform(X[:, 3])


onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
#every column correspond to a specific category 

#Avoiding the Dummy Variable Trap
X = X[:, 1:] #manual approach 
#we do not take the first column at column 0
#this is to make sure bt mostly it is done by default by the library 
#1 Column is removed 


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, /
                    test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting Multi Linear Regression to the Traning Data
from sklearn.linear_model import LinearRegression 
regressor= LinearRegression()
#fit the regression to traning set
regressor.fit(X_train, y_train)

#predicting the Test set results
y_pred= regressor.predict(X_test)
#output predicted based on Test set

#Building the optimal model using Backward Elimination 
import statsmodels.formula.api as sm
#add a new array for the intial constant in y=mx+c*x0
#ss x0 is appened as a array of 1
X = np.append(arr=np.ones(( 50,1)).astype(int), values =X , axis=1)
#50 = lines, 1=column, axis = 1 add columns , 0-rows 
#astype to solve data type error
#the above matrix of one's is being added to the end of the X
#to have it at the begining we put np.ones to arr
#thsi adds coloumn of oen at the begining of the matrix

#variables which have high impact on profit 
X_opt = X[:, [0,1,2,3,4,5]]
#all the possible predictors fitted 
regressor_OLS= sm.OLS(endog = y, exog=X_opt).fit()
#Optimal Least Square =OLS
regressor_OLS.summary()
#to find P values 

#it p>5% we remove it so we remove the variable 
#with the highest p value so we remove x2
#variables which have high impact on profit 
X_opt = X[:, [0,1,3,4,5]]
#all the possible predictors fitted 
regressor_OLS= sm.OLS(endog = y, exog=X_opt).fit()
#Optimal Least Square =OLS
regressor_OLS.summary()
#to find P values


X_opt = X[:, [0,3,4,5]]
regressor_OLS= sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,3,5]]
regressor_OLS= sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()
#so it is r&d and marketing 

#as the p value is 0.060 it is still 
#greater than 5% so 

X_opt = X[:, [0,5]]
regressor_OLS= sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()
#so r&d has the highest significance


