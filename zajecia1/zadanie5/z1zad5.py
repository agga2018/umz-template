#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# wczytanie danych treningowych
os.chdir('/home/aga/uczmas1/umz-template/zajecia1/zadanie5/train')
report=pd.read_csv('in.tsv', sep='\t', names=['price', 'mileage', 'year', 'brand', 'engingeType',	'engineCapacity'])
x_train = report['year'] 
y_train = report['price']

m=5000
b=0
epoki=1000
lr1=0.0000000002
lr2=0.005
N=float(len(y_train))

for i in range (epoki):
	y = (m*x_train)+b
	koszt = sum([data**2 for data in (y_train-y)])/N
	m_gradient = -(1/N)*sum(x_train*(y_train-y))
	b_gradient = -(1/N)*sum(y_train-y)
	m = m - (lr1*m_gradient)
	b = b - (lr2*b_gradient)


# wczytanie danych testowych
os.chdir('/home/aga/uczmas1/umz-template/zajecia1/zadanie5/dev-0')
report2=pd.read_csv('in.tsv', sep='\t', names=['mileage', 'year', 'brand', 'engingeType',	'engineCapacity'])
x_dev=pd.DataFrame(report2, columns=['year'])

# wczytanie danych testowych2
os.chdir('/home/aga/uczmas1/umz-template/zajecia1/zadanie5/test-A')
report3=pd.read_csv('in.tsv', sep='\t', names=['mileage', 'year', 'brand', 'engingeType',	'engineCapacity'])
x_test=pd.DataFrame(report3, columns=['year'])

# plik out w katalogu dev
y_dev = (m*x_dev)+b
os.chdir('/home/aga/uczmas1/umz-template/zajecia1/zadanie5/dev-0')
pd.DataFrame(y_dev).to_csv('out.tsv', sep='\t', index=False, header=False)

# plik out w katalogu test-A 
y_test = (m*x_test)+b
os.chdir('/home/aga/uczmas1/umz-template/zajecia1/zadanie5/test-A')
pd.DataFrame(y_test).to_csv('out.tsv', sep='\t', index=False, header=False)









