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
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/d70c60d7-49e2-4c1f-b03e-426836b071e4)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/26d04f9f-5577-44ac-91a2-66dc2b9f53b0)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/705328e4-e5cb-49f3-857a-d52e450a8c08)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/790bf63a-b578-4c73-b8d2-575a9ce23383)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/c977b6e5-458d-44e0-8d61-18bf5e9f0c27)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/f590b180-8447-4eac-b769-4c73cbbb6fdb)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/f10b8285-5fa0-44c8-b656-c4b4248b6f00)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
