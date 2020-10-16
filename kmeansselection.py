import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from statistics import mode
import os

# Make sure your input file is in a folder called OpenFaceprocessed
# filename : name of the file that contains OpenFace csv output
# num_clusters : user input of number of clusters for kmeans
# dir_name : the name of or path to the directory of the file
# new_dir : the directory which will have the majority cluster file. Make sure this directory is the same directory
# as dir_name

def kmeans_au(filename, num_clusters, dir_name, new_dir):
    filepath = dir_name + "\\" + filename
    open_face_data = pd.read_csv(filepath)
    #print(open_face_data[[' AU01_r']])
    au_scores = open_face_data.loc[:, [" AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r",
                             " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r",
                             " AU15_r", " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r"]]
    kmeans = KMeans(num_clusters, random_state=0)
    kmeans.fit(au_scores)
    result = kmeans.predict(au_scores)
    open_face_data['cluster'] = result
    majority_cluster = mode(result)
    majority_data = open_face_data[open_face_data['cluster'] == majority_cluster]
    majority_data = majority_data.drop(columns=['cluster'])
    newfilename = filename.rstrip(".csv") + "_" + "majority" + ".csv"
    newdirname = dir_name.rstrip("OpenFaceprocessed") + new_dir
    majority_data.to_csv(newdirname + "\\" + newfilename)

# Make sure all the files are in OpenFaceprocessed

dir_name = "C:\\Users\\sahan\\OneDrive\\Documents\\Projects\\HEAL\\OpenFaceprocessed"
files = os.listdir(dir_name)

for file in files:
    kmeans_au(file, 5, dir_name, "kmeansOutput2")
print("Done")

