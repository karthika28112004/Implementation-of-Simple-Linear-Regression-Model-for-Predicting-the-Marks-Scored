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
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/356f8110-4b46-40c3-9c2c-df4de14cd4bf)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/8b06bced-997c-4d9b-977d-1b6bbd068424)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/335984e6-f95a-471f-89c1-ef1c9ac7d346)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/6127e19a-e0e8-485f-86f4-497025fd2cb8)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/ae2d8312-d25d-4f33-b08f-5ea4ec262881)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/ffd9b67d-25b9-441c-a2f6-79fce0f1ef42)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/72a1f60d-84cf-496e-a87e-c184c5ad98e2)
![image](https://github.com/karthika28112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128035087/f4cdc498-e3d8-4138-89dc-a255d847dbee)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
