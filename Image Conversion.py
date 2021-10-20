from PIL import Image
import numpy as np
import os
from pathlib import PurePath
from csv import writer


def ImageFileList(Location, format='.jpg'):
    fileList = []
    for rootloc, directories, files in os.walk(Location, topdown=False):
        for nwfile in files:
            if nwfile.endswith(format):
                PathName = os.path.join(rootloc, nwfile)
                fileList.append(PathName)
    return fileList

for total in range(0,10):
    Chr=str(total)
    link = PurePath('C:/Users/Sreerupu/Desktop/Rupesh/MBA Classes/Semester 3/Subjects/Machine Learning and AI/Project/CNN/Data/MNIST/ActualtrainingSet/trainingSet/',Chr)
    myFileList = ImageFileList(link)
    for file in myFileList:
        img_file = Image.open(file)

        # get original image parameters...
        width, height = img_file.size
        # format = img_file.format
        # mode = img_file.mode
        #
        # # Make image Greyscale
        # img_grey = img_file.convert('L')
        value = np.asarray(img_file.getdata(), dtype=np.int).reshape(-1,height,width)

        value = value.flatten()
        with open("C:/Users/Sreerupu/Desktop/Rupesh/MBA Classes/Semester 3/Subjects/Machine Learning and AI/Project/CNN/Data/Test/Check_Data.csv", 'a') as newimg:
            csv_writer = writer(newimg,lineterminator = '\n')
            csv_writer.writerow(value)