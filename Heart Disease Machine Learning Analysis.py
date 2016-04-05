import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

csv_loc = "C:/Users/dano/Desktop/Heart Disease Data/processed_hungarian_data.csv"

data = pd.read_csv(csv_loc, sep=';')

colours = np.random.rand(50, 50)

area = np.pi * (15 * np.random.rand(50)) ** 2 #0 to 15 point radiuses

def process_array(convert_array):
	new_array = np.array([])
	for i in convert_array:
		if i != '?':
			new_array = np.append(new_array, float(i))
		else:
			new_array = np.append(new_array, 0)
	print "NEW ARRAY:", new_array
	return new_array



blood_press = process_array(data['trestbps'])
ages = process_array(data['age'])

print "Blood Pressure:", blood_press
print "Ages:", ages

plt.scatter(blood_press, ages, s=10, alpha=0.5)
plt.show()


