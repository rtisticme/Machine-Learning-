# Salary based on years of experience

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting Simple Linear Regression to the Traning set 
from sklearn.linear_model import LinearRegression 
regression=LinearRegression()
regression.fit(X_train, y_train)
#this is how we created the simple linear regression model 
#this learned the correlation in the training set

#Predicting the Test set results 
#Vector of prediction
y_predicted = regression.predict(X_test)
#y_test has the real salary and y_predicted has 
#the predicted salary 

#Visualising the Traning set results by plotting
plt.scatter(X_train, y_train, color='red')
#this will plot the real values in red 
plt.plot(X_train, regression.predict(X_train), color= 'blue' )
#this will plot the training set result 
plt.title('Salary vs Experience (Traning Set)')
plt.xlabel('experience')
plt.ylabel('Number of years')
plt.show()
#red line show the real value 
#the blue line show the predicted value 

#Visualising the Test set results by plotting
plt.scatter(X_test, y_test, color='red')
#this will plot the real values in red 
plt.plot(X_train, regression.predict(X_train), color= 'blue' )
#this doesnot change as the value is based on traning set 
#this will plot the training set result 
plt.title('Salary vs Experience (Test)')
plt.xlabel('experience')
plt.ylabel('Number of years')
plt.show()
#red line show the real value 
#the blue line show the predicted value 
