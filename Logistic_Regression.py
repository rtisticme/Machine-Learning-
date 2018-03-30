#Logistic Regression 

#Importing Libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data set
#we want to look at the age and their salary to see if he/she 
#click and purchases a product 
dataset = pd.read_csv('Social_Network_Ads.csv')
X= dataset.iloc[:, [2,3]].values 
#[[]]- converted to a matrix 
Y=dataset.iloc[:, 4].values
#vector

#Splitting the dataset into traning and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state= 0)
classifier.fit(X_train, Y_train)

#Predicting the test set result 
Y_pred = classifier.predict(X_test)

#Making the Confusion Matrix for evaluating the prediction power 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
#passing real and predicted value for comparision 
#88 correct
#12 incorrect

#Visualizing the training set result 
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
#Creating local variable to create a shortcut for replacing xtrain and ytrain 
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
#we prepare the grid with all pixel points
#we take min values of age value -1, so that the points wont be squezeed
#and maximam values of age with 0.01 resolution, same for salary

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('black','white')))
#we apply calssifier on all the pixel points to ceate a contour between blu and red points
#if picture points belong to class 0 it colorize the pont red 
#and if picture point belong to class 1 it will be colorized to green

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(),X2.max())
#Colorizing each pixel pont with black or white
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set ==j,0], X_set[Y_set==j, 1],
                c = ListedColormap(('red','green'))(i), label = j)
#we plot all the red and blue data points    
plt.title('Logistic Regression (Traning set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
#to show the value for red and green 
plt.show()
#there are some red and green points which are the observation point 
#of the traning set
#red points are users who didnot buy 
#green people are the people who buy 
#so young people with low salary didnot buy 
#people who are older and have a high estimated the salary bought th SUV
#Some old people with low estimated salary also bought the SUV 
#it is a linear logistic regression as it is a straight line seperating 
#two conditions, which is called the prediction bnoundary seperator 
#as the users are not lineraly distured it gives soem very incorrect values 

#Visualizing the test set result 
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('black','white')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set ==j,0], X_set[Y_set==j, 1],
                c = ListedColormap(('red','green'))(i), label = j)
plt.title('Logistic Regression (Traning set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
#Black is for no
#White is for yes 