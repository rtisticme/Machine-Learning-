#Decision Tree Regression 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y= dataset.iloc[:, 2].values

#fitting decesion tree to the dataset
from sklearn.tree import DecisionTreeRegressor 
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

#predicting the values based on decision tree 
predict_y = regressor.predict(6.5)

#visualizing the result 
#the predictions must be constant between 
#but this a different value but decision tree works on information entropy
#so the value must be non-continous ladder shape
#the salary value is the average s
plt.scatter(X,Y, color= 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth Vs Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualing in higher resolution 
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y, color= 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth Vs Bluff(Decision Tree)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
