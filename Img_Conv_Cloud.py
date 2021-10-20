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

for total in range(2,5):
    Chr=str(total)
    link = PurePath('gs://sparklearningnew/MNIST Dataset/',Chr)
    myFileList = ImageFileList(link)
    for file in myFileList:
        img_file = Image.open(file)

        width, height = img_file.size
        value = np.asarray(img_file.getdata(), dtype=np.int).reshape(-1,height,width)

        value = value.flatten()
        with open("gs://sparklearningnew/MNIST Dataset/Test.csv", 'a') as newimg:
            csv_writer = writer(newimg,lineterminator = '\n')
            csv_writer.writerow(value)