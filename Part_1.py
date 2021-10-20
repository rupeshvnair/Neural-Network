#Data Preparation
import pandas as pd
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
#import tensorflow as tf

#Loading the data
train = pd.read_csv("C:/Users/Sreerupu/Desktop/Rupesh/MBA Classes/Semester 3/Subjects/Machine Learning and AI/Project/CNN/Data/Kaggle/train_cp.csv")
test = pd.read_csv("C:/Users/Sreerupu/Desktop/Rupesh/MBA Classes/Semester 3/Subjects/Machine Learning and AI/Project/CNN/Data/Kaggle/test.csv")

Y_labels = train["label"]
X_train = train.drop(labels="label",axis = 1)

del train
#g = sns.countplot(Y_train)
#g.set(xlabel ='Numbers',ylabel = 'Count')
#plt.show()

#Checking Null values
print(X_train.isnull().any().describe())
print(test.isnull().any().describe())


#Normalization
X_train = X_train/255.0
test = test/255.0

#Reshaping
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

print(X_train)
X_train= X_train.flatten()


#Label encoding
Y_train = to_categorical(Y_labels, num_classes=10)

#Split training and validation set
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size=0.1, random_state= 2)
print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)
print("No. of GPU is ",len(tf.config.experimental.list_physical_devices('GPU')))
