#SVR

#Importing libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing dataset 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:, 2].values

#feature scaling is nedded in SVR
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
Y = sc_y.fit_transform(Y)


#fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
#kernel:- linear,(non-linear) poly, rbf, sigmoid 
regressor.fit(X, Y)

#predicting the result
y_pred = regressor.predict(6.5)

#visualizing 
plt.scatter(X,Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth Vs Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()