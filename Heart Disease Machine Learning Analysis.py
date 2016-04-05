import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LinearRegression

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
print processed_data

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pca_2 = PCA(2)

plot_columns = pca_2.fit_transform(processed_data)

plt.scatter(plot_columns[:,0],plot_columns[:,1])

"""
ages = ages.reshape(len(ages),1)
blood_press = blood_press.reshape(len(blood_press),1)

regression = LinearRegression()
regression.fit(ages, blood_press)

colour_one = "b"

for age, bp in zip(ages, blood_press):
	plt.scatter(ages, blood_press, color=colour_one)
"""
plt.title("Heart Disease")

plt.grid(alpha=0.4)

try:
	plt.plot(blood_press, reg.predict(blood_press))
except NameError:
	print "Regression NameError"

plt.show()


