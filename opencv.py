import numpy as np
import cv2
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# File path
path = 'C:/Users/sahan/Downloads/OpenFace_2.2.0_win_x64/OpenFace_2.2.0_win_x64/processed'
path_names = ['S7_7.5lbs_Mid_M2T_6_aligned', 'S26_7.5lbs_Mid_G2T_1_aligned',
             'S10_35lbs_Mid_G2M_9_aligned', 'S13_25lbs_Mid_G2M_7_aligned',
             'S31_35lbs_Mid_G2M_4_aligned', 'S13_25lbs_Mid_G2M_7_aligned']
kmeanspath = "C:/Users/sahan/OneDrive/Documents/Projects/HEAL/kmeansOutput2"
# images = os.listdir(path)
i = 0
j = 0
X = []
y =[]
for j in range(len(path_names)):
    dir_path = path + '/' + path_names[j]
    images = os.listdir(dir_path)
    majority_path = kmeanspath + '/' + path_names[j].rstrip('aligned') + 'majority.csv'
    majority = pd.read_csv(majority_path)
    majority = majority.drop(majority.columns[[0]], axis=1)
    # print(images)
    for i in range(len(images)):
        # print("hey")
        image_path = dir_path + "/" + images[i]
        frameno = images[i].split('_')
        frame = frameno[3].rstrip(".bmp")
        frame = int(frame)
        # print(frame)
        # print(image_path)
        # read every image in the file and add it to the X_train
        if frame in list(majority.frame):
            data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            X.append(data)

            argument = dir_path.split('/')
            # print(argument)
            weights = argument[7].split('_')
            weight = weights[1].rstrip("lbs")
            weight = float(weight)

            # find the weights and accordingly decide the label
            if weight == 7.5:
                y.append(0)
                # y_train.append('light')
            elif weight == 25:
                y.append(1)
                # y_train.append('medium')
            else:
                y.append(2)
                # y_train.append('heavy')


        #i = i + 1

    #j = j + 1
    #print(j)



# cv2.imshow("test",X_train[0])
# cv2.waitKey(0)
print(np.shape(X))
print(np.shape(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
shape = np.shape(X_train)
print(np.shape(X_test))
# print(np.shape(X_train))
# print(np.shape(y_train))
# X_randomtrain = []
# Y_randomtrain = []
#
# x = random.randint(502, size=(100))
# for i in x:
#     X_randomtrain

#cv2.imshow("test", X_train[6])
#cv2.waitKey(0)
Y_train = np_utils.to_categorical(y_train, 3)
Y_test = np_utils.to_categorical(y_test, 3)

X_train = np.array(X_train)
X_train = X_train.astype('float32')
X_train /= 255

X_test= np.array(X_test)
X_test = X_test.astype('float32')
X_test /= 255


print(np.shape(X_train))
print(np.shape(Y_train))
model = Sequential()

# input layer
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(112, 112, 3)))
# print(model.output_shape)
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# output layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(Y_train.dtype.name)

# Fit model on training data
model.fit(X_train, Y_train,
          batch_size=1, epochs=10, verbose=1)

model.summary()


# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
