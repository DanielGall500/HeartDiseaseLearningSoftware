import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

csv_loc = "C:/Users/dano/Desktop/Heart Disease Data/processed_hungarian_data.csv"

data = pd.read_csv(csv_loc, sep=';')

colours = np.random.rand(50, 50, 0)

area = np.pi * (15 * np.random.rand(50)) ** 2 #0 to 15 point radiuses

def process_array(convert_array):
	new_array = np.array([])
	for i in convert_array:
		if i != '?':
			np.append(new_array, float(i))
	return new_array



blood_press = process_array(data['trestbps'])
ages = process_array(data['age'])

plt.scatter(blood_press, ages, s=100, alpha=0.5)
plt.show()


