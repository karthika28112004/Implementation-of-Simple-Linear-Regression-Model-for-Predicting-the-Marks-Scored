# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: E.Karthika
RegisterNumber: 212222040072

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/fd0bd50d-1fef-486f-90be-4924b379a608)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/e3297189-6b0f-4ed8-9926-ef8282fde70f)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/c244284c-72ed-4540-8090-c6fe77980006)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/cebf8b29-7828-40d2-aeda-eb251916f7c1)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/e62303bd-f373-49c5-9480-64561640a280)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/5886364a-0bbe-4fb8-be6b-c5aa7b8c077b)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/5d8aacd7-efbf-41a8-a2a5-a478e4e5cdc7)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
