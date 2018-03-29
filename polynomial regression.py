#Polynomial regression 

#Importing Libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[:, 1:2].values
#1:2 is to create a matrix and not a array
#always create x as matrix and y as vector 
Y= dataset.iloc[:,2].values
#we onlyk require 2 column

#as we have a small data set we donot have to split it in to
#traning set and test set

#Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Fitting polynomial linear regression to dataset 
from sklearn.preprocessing import PolynomialFeatures
#poly_reg = PolynomialFeatures(degree=2)
#this add x1^2 column 
#we can add a degree to make the observation better
#poly_reg = PolynomialFeatures(degree=3)
#this add x1^2, x1^3 column 
poly_reg = PolynomialFeatures(degree=4)
#using one more degree gives us a perfect model 
#this add x1^2 column 
X_poly= poly_reg.fit_transform(X)
#this converts x to x_poly adding a addtional x1^2 variable
#this also automatically adds columns of 1 as a constant b0
lin_reg2 = LinearRegression()
#this will be to fit the mutiple linear regression to x2 
lin_reg2.fit(X_poly, Y)

#visualising the Linear regression results
plt.scatter(X,Y, color= 'red')
#giving x and  y coordinates 
plt.plot(X, lin_reg.predict(X), color= 'blue')
#predict value of Y form X by linear regression model 
plt.title('TruthvsBluff (Linear Regression)')
plt.xlabel('Positon Level')
plt.ylabel('Salary')
plt.show()
#this gives a very inaccurate result
#so we need to create a non-linear model so that it wil match 
#the y observation 

#visualising the Polynomial Linear regression results
#to create a contious plot 
#without this it creates straight line in different places
X_grid = np.arange(min(X), max(X), 0.1)
#lower bound and upper bound that increases by 0.1
#this gives us a vector we require a matrix 
X_grid = X_grid.reshape((len(X_grid),1))
#its creates a matrix of 1 column with 90
#no of lines= number of line in x grid 
#has 90 lines incremented by 1

plt.scatter(X,Y, color= 'red')
#giving x and  y coordinates 
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color= 'blue')
#donot use x poly as it was already defined as existing matrix of x
#we need to add the polynomial expression to the the prediction 
#model
#predict value of Y form X by linear regression model 
plt.title('TruthvsBluff (Polynomial Regression)')
plt.xlabel('Positon Level')
plt.ylabel('Salary')
plt.show()
#this gives a very inaccurate result
#so we need to create a non-linear model so that it wil match 
#the y observation 

#Predicting a new result with Linear Regression 
lin_reg.predict(6.5)
#predicting a salary for the level 6.5 (Human Resources)

#Predicting a new result with Polynomial Linear Regression 
lin_reg2.predict(poly_reg.fit_transform(6.5))
#predicting a salary for the level 6.5 (Human Resources)