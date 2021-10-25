from PIL import Image
import numpy as np
import os
from pathlib import PurePath
from csv import writer
import pandas as pd
import numpy as np
import spark
from csv import writer
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

!wget "https://raw.githubusercontent.com/rupeshvnair/BigDataMachinelearning/main/archive.zip"

!unzip "/etc/jupyter/symlinks_for_jupyterlab_widgets/Local Disk/archive.zip"



def ImageFileList(Location, format='.jpg'):
    fileList = []
    for rootloc, directories, files in os.walk(Location, topdown=False):
        for nwfile in files:
            if nwfile.endswith(format):
                PathName = os.path.join(Location, nwfile)
                fileList.append(PathName)
    return fileList



#For Training Dataset
for total in range(0,10):
    Chr=str(total)
    link = PurePath('/etc/jupyter/symlinks_for_jupyterlab_widgets/Local Disk/trainingSet/trainingSet/',Chr)
    myFileList = ImageFileList(link)
    for file in myFileList:
        img_file = Image.open(file)
        width, height = img_file.size
        value = np.asarray(img_file.getdata(), dtype=np.int).reshape(-1,height,width)

        value = value.flatten()
        value = np.append(total,value)#appending the labels to the pixel data
        with open("/etc/jupyter/symlinks_for_jupyterlab_widgets/Local Disk/Results/Check_New_Data.csv", 'a') as newimg:
            csv_writer = writer(newimg,lineterminator = '\n')
            csv_writer.writerow(value)



#For Test Dataset
link = PurePath('/etc/jupyter/symlinks_for_jupyterlab_widgets/Local Disk/testSet/testSet/',Chr)
myFileList = ImageFileList(link)
for file in myFileList:
    img_file = Image.open(file)
    width, height = img_file.size
    value = np.asarray(img_file.getdata(), dtype=np.int).reshape(-1,height,width)

    value = value.flatten()
    with open("/etc/jupyter/symlinks_for_jupyterlab_widgets/Local Disk/Results/Pred_Data.csv", 'a') as newimg:
        csv_writer = writer(newimg,lineterminator = '\n')
        csv_writer.writerow(value)



from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .getOrCreate()


train = spark.read.format("csv").option("header", "false").load("gs://sparklearningnew/notebooks/jupyter/Check_New_Data.csv")
test = spark.read.format("csv").option("header", "false").load("gs://sparklearningnew/notebooks/jupyter/Pred_Data.csv")


from pyspark.sql.types import IntegerType

for values in train.columns:
    train=train.withColumn(values, train[values].cast(IntegerType()))


for values in test.columns:
    test=test.withColumn(values, test[values].cast(IntegerType()))


labels = spark.sql("Select 0 from df_sql")

Digit_training = train.toPandas()
Digit_Prediction = test.toPandas()


Digit_labels = Digit_training[0]
Digit_train_Pixel_data = Digit_training.drop(labels=0,axis = 1)


g = sns.countplot(Digit_labels)
g.set(xlabel ='Numbers',ylabel = 'Count')
plt.show()


print(Digit_train_Pixel_data.isnull().any().describe())
print(Digit_Prediction.isnull().any().describe())


!pip install tensorflow

#Importing Modules for Machine Learning

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
sns.set(style = 'white',context = 'notebook')



#Normalization
train_set = Digit_train_Pixel_data/255.0
test_set = Digit_Prediction/255.0

#Reshaping
train_set = train_set.values.reshape(-1,28,28,1)
test_set = test_set.values.reshape(-1,28,28,1)


#Label encoding
Digit_labels = to_categorical(Digit_labels, num_classes=10)

#Split training and validation set
train_set_1, train_eval, Digit_labels_1, Digit_labels_2 = train_test_split(train_set,Digit_labels,test_size=0.1, random_state= 2)



print(train_set_1.shape, train_eval.shape, Digit_labels_1.shape, Digit_labels_2.shape)
plt.imshow(train_set_1[5][:,:,0])
plt.show()


#Model defining
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))



# Define the optimizer
optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer
learning_rate_modification = ReduceLROnPlateau(monitor='val_accuracy',
                                               patience=3,
                                               verbose=1,
                                               factor=0.5,
                                               min_lr=0.00001)



epochs = 30
batch_size = 95


ImgGen = ImageDataGenerator(
    #featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.1, # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images


ImgGen.fit(train_set_1)
# Fit the model
history = model.fit_generator(ImgGen.flow(train_set_1,Digit_labels_1, batch_size=batch_size),
                              epochs = epochs, validation_data = (train_eval, Digit_labels_2),
                              verbose = 2, callbacks=[learning_rate_reduction])



fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# Check confusion matrix

def confusion_matrix_plot(tcm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.tcm.Blues):
    plt.imshow(tcm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = tcm.max() / 2.
    for i, j in itertools.product(range(tcm.shape[0]), range(tcm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted_label')


# Predict the values from the validation dataset
Y_pred = model.predict(train_eval)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred,axis = 1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(Digit_labels_2,axis = 1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# print(confusion_mtx.max())
# plot the confusion matrix
confusion_matrix_plot(confusion_mtx, classes = range(10))


# predict results
results = model.predict(test_set)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Model_Result_Label")