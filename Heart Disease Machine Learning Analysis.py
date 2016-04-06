import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn import svm

csv_loc = "C:/Users/dano/Desktop/Heart Disease Data/processed_hungarian_data.csv"

data = pd.read_csv(csv_loc, sep=';')
data = data.replace('?',0)

colours = np.random.rand(50, 50)

area = np.pi * (15 * np.random.rand(50)) ** 2 #0 to 15 point radiuses

def process_array(convert_array):
	new_array = np.array([])
	for i in convert_array:
		if i != '?':
			value = np.array([float(i)])
			new_array = np.concatenate((new_array,value))
		else:
			new_array = np.concatenate((new_array, [0])) #NEED TO DELETE BEFORE ML
	return new_array

def process_whole_array(dataframe):
	for row in dataframe:
		dataframe[row].apply(lambda x: float(x))
	return dataframe
			
"""
blood_press = process_array(data['trestbps'])
ages = process_array(data['age'])
genders = process_array(data['gender'])
chest_pain = process_array(data['cp'])
cholestoral = process_array(data['chol'])
blood_sugar = process_array(data['fbs'])
resting_ecg = process_array(data['restecg'])
max_heartrate = process_array(data['thalach'])
exercise_angina = process_array(data['exang'])
st_depression = process_array(data['oldpeak'])
"""
processed_data = process_whole_array(data)

feature_names = ['age','gender','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']
label_names = ['num']

data_features = processed_data[feature_names]
data_labels = processed_data['num']

label_list = []
for i in data_labels:
	label_list.append(i)
	print "AP:", i

from sklearn.decomposition import PCA

pca_2 = PCA(2)

plot_columns = pca_2.fit_transform(processed_data)

#plt.scatter(plot_columns[:,0],plot_columns[:,1])

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(data_features, label_list, test_size=0.33, random_state=42)

clf = svm.SVC()

print features_train
print labels_train

clf.fit(features_train,labels_train)

predictions = clf.predict(features_test)

from sklearn.metrics import accuracy_score
print "Accuracy Score:", accuracy_score(labels_test, predictions)


plt.title("Heart Disease")

plt.grid(alpha=0.4)

plt.show()


