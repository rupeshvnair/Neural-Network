import pandas as pd
import numpy as np
from csv import writer
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
from PIL import Image
import os
from pathlib import PurePath

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
from sklearn.decomposition import PCA


#to get the MNIST Image data zip file to the google cluster
!wget "https://raw.githubusercontent.com/rupeshvnair/BigDataMachinelearning/main/archive.zip"

#Unzipping the MNIST Image data on the google cluster
!unzip "/etc/jupyter/symlinks_for_jupyterlab_widgets/Local Disk/archive.zip"


#Defining a function to make a list of the locations of all the images in a particular file
def ImageFileList(Location, format='.jpg'):
    fileList = []
    for rootloc, directories, files in os.walk(Location, topdown=False):
        for nwfile in files:
            if nwfile.endswith(format):
                PathName = os.path.join(Location, nwfile)
                fileList.append(PathName)
    return fileList



#For Training Dataset - The training images are converted to pixel data csv and appended with labels
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
        with open("/etc/jupyter/symlinks_for_jupyterlab_widgets/Local Disk/Results/Train_Data.csv", 'a') as newimg:
            csv_writer = writer(newimg,lineterminator = '\n')
            csv_writer.writerow(value)



#For Prediction Dataset - The prediction images are converted to pixel data csv
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


#Loading both the training and prediction pixel data to a python dataframe
Digit_training = pd.read_csv("/content/drive/MyDrive/Coding/Dataset/Train_Data.csv",header=None)
Digit_Prediction = pd.read_csv("/content/drive/MyDrive/Coding/Dataset/Pred_Data.csv",header=None)

#Splitting the training data into pixel information and labels
Digit_labels = Digit_training[0]
Digit_train_Pixel_data = Digit_training.drop(labels=0,axis = 1)


#Checking Class Imbalance in the data
g = sns.countplot(Digit_labels)
g.set(xlabel ='Numbers',ylabel = 'Count')
plt.show()

#Checking the training and prediction data for null values
print(Digit_train_Pixel_data.isnull().any().describe())
print(Digit_Prediction.isnull().any().describe())

#Normalizing the pixel data of training dataset and prediction dataset
train_set = Digit_train_Pixel_data/255.0
test_set = Digit_Prediction/255.0

#Label encoding for the labels of Training Dataset
Digit_labels = to_categorical(Digit_labels, num_classes=10)


#Performing PCA on the training dataset and reducing the number of features required
pca_40 = PCA(n_components=40)
pca_40_reduced = pca_40.fit_transform(train_set)
pca_40_recovered = pca_40.inverse_transform(pca_40_reduced)
plt.grid()
plt.plot(np.cumsum(pca_40.explained_variance_ratio_ * 100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')

#Reshaping
train_set = pca_40_recovered.reshape(-1,28,28,1)
test_set = test_set.values.reshape(-1,28,28,1)

#Split training dataset to train set and validation set so that with the help of validation set the model can predict its efficiency
train_set_1, train_eval, Digit_labels_1, Digit_labels_2 = train_test_split(train_set,Digit_labels,test_size=0.1, random_state= 2)


#Defining the Machine Learning Model using Keras Sequential api
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

# Define the optimizer which is used for back propagation
optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model with the optimizer
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


#the number of times the model backpropagates and learns is specified
epochs = 30
batch_size = 85

#Defining the Image Data generator so that we can reduce overfitting and increase the training data
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


# Fit the model with the training data, generator data, validation data, training labels and validation labels
history = model.fit_generator(ImgGen.flow(train_set_1,Digit_labels_1, batch_size=batch_size),
                              epochs = epochs, validation_data = (train_eval, Digit_labels_2),
                              verbose = 2, callbacks=[learning_rate_reduction])


#Plotting the training loss to validation loss
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)


#Plotting the training accuracy to validation accuracy
ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


#
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
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
print(confusion_mtx.max())
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))


# Display some error results

# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)
# print(errors)
Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_cat_errors = Y_pred[errors]

Y_true_classes_errors = Y_true[errors]

train_eval_errors = train_eval[errors]


plt.imshow((train_eval_errors[10]).reshape((28,28)))
plt.title("Predicted label :{}\nTrue label :{}".format(Y_pred_classes_errors[10],Y_true_classes_errors[10]))



# Display some correct results

# Errors are difference between predicted labels and true labels
correct_pred = (Y_pred_classes - Y_true == 0)
# print(errors)
Y_pred_classes_correct = Y_pred_classes[correct_pred]

Y_true_classes_correct = Y_true[correct_pred]

train_eval_correct = train_eval[correct_pred]


plt.imshow((train_eval_correct[80]).reshape((28,28)))
plt.title("Predicted label :{}\nTrue label :{}".format(Y_pred_classes_correct[80],Y_true_classes_correct[80]))




