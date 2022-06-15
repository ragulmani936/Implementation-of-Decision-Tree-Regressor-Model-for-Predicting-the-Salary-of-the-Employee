# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:Ragul M 
RegisterNumber:212221230080  
*/
```
~~~
import pandas as pd
d=pd.read_csv("Salary.csv")
d.head()
d.info()
d.isnull().sum()

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
d["Position"] = l.fit_transform(d["Position"])
d.head()

x = d[["Position","Level"]]
y = d["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()

dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2 = metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
~~~
## Output:
## Data head:
![pic 1](https://github.com/ragulmani936/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/blob/main/data%20head.png)
## Data info:
![pic 2](https://github.com/ragulmani936/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/blob/main/data%20info.png)
## Data isnull:
![pic 3](https://github.com/ragulmani936/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/blob/main/data%20isnull.png)
## Label Encoder:
![pic 4](https://github.com/ragulmani936/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/blob/main/label%20encoder.png)
## MSE:
![pic 5](https://github.com/ragulmani936/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/blob/main/metrics.png)
## R2:
![pic 6]()
## Predict:
![pic 7]()
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
