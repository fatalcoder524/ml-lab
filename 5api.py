import csv
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
data = pd.read_csv('pima_indian_diabetes.csv')
x = np.array(data.iloc[:,0:-1])
y = np.array(data.iloc[:,-1])  
print(data.head())
model = GaussianNB()
model.fit(x,y)
predicted= model.predict([[6,149,78,35,0,34,0.625,54]])
print("Predicted Value:", predicted)
