import numpy as np
import cv2
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


dir_name = "D:\\processed"
X = []
y = []
files = os.listdir(dir_name)
count = 0
for file in files:
    if file.endswith(".csv"):
        count = count+1
        print(file)
        filepath = dir_name + "\\" + file
        open_face_data = pd.read_csv(filepath)
        au_scores = open_face_data.loc[:, ["frame", " confidence", " AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r",
                                           " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r",
                                           " AU15_r", " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r"]]
        confident = au_scores[au_scores[" confidence"] > 0.8]
        if confident.shape[0] > 35:
            samples = confident.sample(n=35).sort_values(by="frame")
            if count == 1:
                X = samples.to_numpy()
            else:
                print(samples.to_numpy().shape)
                print(X.shape)
                X = np.dstack([X, samples.to_numpy()])









print(count)


