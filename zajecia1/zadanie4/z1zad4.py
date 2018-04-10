#!/usr/bin/python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt

# wczytanie danych treningowych
os.chdir('/home/aga/uczmas1/umz-template/zajecia1/zadanie4/train')
report=pd.read_csv('in.tsv', sep='\t', names=['price', 'mileage', 'year', 'brand', 'engingeType',	'engineCapacity'])

report_cleared = report[report.year>1975]

x_train = pd.DataFrame(report_cleared, columns=['mileage', 'year', 'engingeType', 'engineCapacity', 'brand'])
y_train = report_cleared['price']
y_train = pd.DataFrame(y_train)

# wczytanie danych testowych
os.chdir('/home/aga/uczmas1/umz-template/zajecia1/zadanie4/dev-0')
report2=pd.read_csv('in.tsv', sep='\t', names=['mileage', 'year', 'brand', 'engingeType',	'engineCapacity'])
x_dev=pd.DataFrame(report2, columns=['mileage', 'year', 'engingeType', 'engineCapacity', 'brand'])

# wczytanie danych testowych2
os.chdir('/home/aga/uczmas1/umz-template/zajecia1/zadanie4/test-A')
report3=pd.read_csv('in.tsv', sep='\t', names=['mileage', 'year', 'brand', 'engingeType',	'engineCapacity'])
x_test=pd.DataFrame(report3, columns=['mileage', 'year', 'engingeType', 'engineCapacity', 'brand'])

# label encoder
le=LabelEncoder()
columns=['engingeType', 'brand']
for col in columns:
		data = pd.concat([x_train[col],x_dev[col],x_test[col]]) 
		le.fit(data.values)
		x_train[col]=le.transform(x_train[col])
		x_dev[col]=le.transform(x_dev[col])
		x_test[col]=le.transform(x_test[col])

# one hot encoder
enc = OneHotEncoder()
x_train_enc = x_train
x_dev_enc = x_dev
x_test_enc = x_test
columns=['engingeType', 'brand']
for col in columns:
	data=x_train[[col]]
	enc.fit(data)
	temp=enc.transform(x_train[[col]])
	temp=pd.DataFrame(temp.toarray(), columns=[(col+"_"+str(i)) for i in data[col].value_counts().index])
	temp=temp.set_index(x_train.index.values)
	x_train_enc=pd.concat([x_train_enc, temp], axis=1)
	temp=enc.transform(x_dev[[col]])
	temp=pd.DataFrame(temp.toarray(), columns=[(col+"_"+str(i)) for i in data[col].value_counts().index])
	temp=temp.set_index(x_dev.index.values)
	x_dev_enc=pd.concat([x_dev_enc, temp], axis=1)
	temp=enc.transform(x_test[[col]])
	temp=pd.DataFrame(temp.toarray(), columns=[(col+"_"+str(i)) for i in data[col].value_counts().index])
	temp=temp.set_index(x_test.index.values)
	x_test_enc=pd.concat([x_test_enc, temp], axis=1)

x_train_enc.drop(['engingeType', 'brand'], inplace=True, axis=1)
x_dev_enc.drop(['engingeType', 'brand'], inplace=True, axis=1)
x_test_enc.drop(['engingeType', 'brand'], inplace=True, axis=1)

# model 
reg=linear_model.LinearRegression()
reg.fit(x_train_enc, y_train)

y_dev_predict = reg.predict(x_dev_enc)
y_test_predict = reg.predict(x_test_enc)
print(reg.coef_)
print(reg.intercept_)

# plik out w katalogu dev
os.chdir('/home/aga/uczmas1/umz-template/zajecia1/zadanie4/dev-0')
pd.DataFrame(y_dev_predict).to_csv('out.tsv', sep='\t', index=False, header=False)

# plik out w katalogu test-A 
os.chdir('/home/aga/uczmas1/umz-template/zajecia1/zadanie4/test-A')
pd.DataFrame(y_test_predict).to_csv('out.tsv', sep='\t', index=False, header=False)

