import numpy as np
import cv2
import os

# File path
path = 'C:/Users/sahan/Downloads/OpenFace_2.2.0_win_x64/OpenFace_2.2.0_win_x64/processed'
path_names = ['S7_7.5lbs_Mid_M2T_6_aligned', 'S26_7.5lbs_Mid_G2T_1_aligned',
             'S10_35lbs_Mid_G2M_9_aligned', 'S13_25lbs_Mid_G2M_7_aligned',
             'S31_35lbs_Mid_G2M_4_aligned', 'S13_25lbs_Mid_G2M_7_aligned']
# images = os.listdir(path)
i = 0
j = 0
x_train = []
y_train = []
for j in range(len(path_names)):
    dir_path = path + '/' + path_names[j]
    images = os.listdir(dir_path)
    # print(images)
    for i in range(len(images)):
        # print("hey")
        image_path = dir_path + "/" + images[i]
        # print(image_path)
        # read every image in the file and add it to the X_train
        data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        x_train.append(data)

        argument = dir_path.split('/')
        # print(argument)
        weights = argument[7].split('_')
        weight = weights[1].rstrip("lbs")
        weight = float(weight)

        # find the weights and accordingly decide the label
        if weight == 7.5:
            y_train.append('light')
        elif weight == 25:
            y_train.append('medium')
            print("hey")
        else:
            y_train.append('heavy')

        #i = i + 1

    #j = j + 1
    #print(j)



# cv2.imshow("test",x_train[0])
# cv2.waitKey(0)
shape = np.shape(x_train)
print(shape)
print(np.shape(x_train))
print(y_train)
# X_randomtrain = []
# Y_randomtrain = []
#
# x = random.randint(502, size=(100))
# for i in x:
#     X_randomtrain