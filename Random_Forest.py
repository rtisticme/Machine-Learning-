#Decision Tree Regression 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y= dataset.iloc[:, 2].values

#Fitting Random Forest to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state=0)
#default random forest is 10 
regressor.fit(X,Y)

#Prediction 
#predicting the values based on decision tree 
predict_y = regressor.predict(6.5)

#Visualing in higher resolution 
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y, color= 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth Vs Bluff(Decision Tree)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#for 10 tree in each level 10 votes were taken and many different 
#averages we calculated to find a accurate value 
#as each tree gives a prediction and all prediction form 
#10 tree is averaged 
#as tree is increased the better the prediction 
#with 300 it gave a value same as the real value 