#SVR (support vector regression)

#Importing libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
#selects column 1
Y = dataset.iloc[:, 2:3].values
#selects column 2
#2:3 is very important or it gives an error

 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_y.fit_transform(Y)
#If feature scaling is not done it gives a straight horizontal 
#line in prediction 

#fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
#kernel:- linear,(non-linear) poly, rbf, sigmoid 
regressor.fit(X, Y)

#predicting the result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
#6.5 is not scaled for prediction so we pass sc_X
#tranform method expects a array 
#so we need to transform the 6.5 numeric value to a array
#np.array is used 
#[]-vector of one element 
#[[]]-array of 1 line 1 column 
#visualizing 
#this will give the scaled version of salary 
#so to obtain the original scale we need to apply 
#inverse transfrom to scaled value of y 

plt.scatter(X,Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth Vs Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#SVR model doesnot fit for the ceo as the data point is very far 
#and sees it as a outlier