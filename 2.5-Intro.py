import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix as scm
df1 = pd.read_csv('C:/Users/Sreerupu/Desktop/Rupesh/MBA Classes/Semester 3/Subjects/Machine Learning and AI/Project/CNN/Data/arrhythmia.data',header=None)
#print(df1.head())

data = df1[[0,1,2,3,4,5]]
data.columns=['age','sex','height','weight','QRS Duration','P-R Interval']

# print(data)
plt.rcParams['figure.figsize']= [15,15]
data.hist()
plt.show()

scm(data)
plt.show()

