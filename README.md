# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Collect the employee dataset containing features like experience, role, and performance along with salary as the target variable.
2. Preprocess the data by handling missing values and splitting it into training and testing sets.
3. Initialize the Decision Tree Regressor model and choose an appropriate criterion (such as squared error).
4. Train the model using the training data to learn decision rules for salary prediction.
5. Test the model on unseen data and evaluate performance using metrics like Mean Squared Error (MSE).

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Jayasri L
RegisterNumber: 212224040136
*/
```

```
import pandas as pd
data=pd.read_csv("/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
r2=metrics.r2_score(y_test,y_pred)
```

## Output:
<img width="1349" height="626" alt="image" src="https://github.com/user-attachments/assets/d0c35154-381a-4c5f-a298-8f3bd2a4d548" />

<img width="1276" height="535" alt="image" src="https://github.com/user-attachments/assets/68871341-fd83-4373-9683-0d1b10fba4de" />

<img width="1208" height="626" alt="image" src="https://github.com/user-attachments/assets/8769d6e8-95b2-4483-bc1a-75e314a999a1" />

<img width="1162" height="290" alt="image" src="https://github.com/user-attachments/assets/0e00b318-0410-4827-9729-b8b8cadfdf32" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
