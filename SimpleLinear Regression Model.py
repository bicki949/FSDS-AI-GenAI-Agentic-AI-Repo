import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"D:\Download\ML-DATA\Salary_Data.csv")

x=dataset.iloc[:,:1]
y=dataset.iloc[:,-1]

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=0
)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train , y_train)


y_pred = regressor.predict(x_test)

comparision = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})
print(comparision)


plt.scatter(x_test,y_test,color='Red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary of based on experience')
plt.xlabel('Experience')
plt.ylable('Salary')
plt.show()




#predict future
m_coef = regressor.coef_
print(m_coef)


c_intercept =regressor.intercept_
print(c_intercept)

y_12 = m_coef *12 + c_intercept
print(y_12)


y_20 = m_coef *20 + c_intercept
print(y_20)




#BIAS
bias=regressor.score(x_train,y_train)
print(bias)
#VARIANCE
variance=regressor.score(x_test,y_test)
print(variance)
# bias-94,variance-98 (best fitted model)