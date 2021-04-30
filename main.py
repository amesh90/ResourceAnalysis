import pandas as pd 
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean

from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
import smogn
from sklearn.metrics import mean_squared_error, make_scorer, explained_variance_score

# Models
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

debug = 1
do_plots = 1
do_training = 0

def plot_graphs(data):



	f = plt.figure()
	f.clear()

	n, bins, patches = plt.hist(x=data['K'], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel('K ')
	plt.ylabel('Frequency')
	plt.xticks([10,100,200])
	plt.title('Histogram of different K values')
	
	plt.savefig("histogram_K")
	plt.close(f)


	f = plt.figure()
	f.clear()

	n, bins, patches = plt.hist(x=data['PQ'], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel('PQ ')
	plt.ylabel('Frequency')
	plt.xticks([1,2],['IndexIVFPQ', 'IndexIVFPQ+R'])
	plt.title('Histogram of different PQ')
	
	plt.savefig("histogram_PQ")
	plt.close(f)


	f = plt.figure()
	f.clear()

	n, bins, patches = plt.hist(x=data['Total_time'], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel('Total Search Time ')
	plt.ylabel('Frequency')
	plt.xscale('log')

	plt.title('Histogram of total search time')
	
	plt.savefig("histogram_totalsearchtime")
	plt.close(f)

	f = plt.figure()
	f.clear()

	n, bins, patches = plt.hist(x=data['Quantization_time'], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel('Quantization Search Time ')
	plt.ylabel('Frequency')
	plt.xscale('log')

	plt.title('Histogram of quantization search time')
	
	plt.savefig("histogram_quantizationsearchtime")
	plt.close(f)

def main_logic(data):

	# drop quantization column
	df = data.drop('Quantization_time', axis=1)

	X = df.iloc[:,:-1]
	y = df.iloc[:,-1]

	

	# split training, testing set
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)

	if debug == 1:
		# summarize class distribution
		print("Before undersampling: ", Counter(y_train))

	# define undersampling strategy
	undersample = RandomUnderSampler(sampling_strategy='majority')

	# fit and apply the transform
	X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)

	if debug == 1:
		# summarize class distribution
		print("After undersampling: ", Counter(y_train_under))

	# Step 1:
	# handle imbalanced data

	# step 2:
	# One hot encoding for K, PQ values
	# Get one hot encoding of columns B
	# one_hot = pd.get_dummies(df['B'])
	# # Drop column B as it is now encoded
	# df = df.drop('B',axis = 1)
	# # Join the encoded df
	# df = df.join(one_hot)


	# step 3: 
	# divide the training and testing set
	

	# step 4:
	# do the cross validation loop

	# prediction and print the accuracy

def main_logic2(data):

	# drop quantization column
	df = data.drop('Quantization_time', axis=1)

	# One hot encoding for K, PQ values
	
	one_hot = pd.get_dummies(df['PQ'])
	df = df.drop('PQ',axis = 1)
	df = df.join(one_hot)

	one_hot = pd.get_dummies(df['K'])
	df = df.drop('K',axis = 1)
	df = df.join(one_hot)

	

	# split training, testing set
	# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)


	# Over sampling
	# os_df = smogn.smoter(
	    
	#     data = df, 
	#     y = "Total_time"
	# )
	# with open('os_df.pkl', 'wb') as f:
	# 			pickle.dump(os_df, f)

	# os = SMOTE(sampling_strategy=0.1)
	# os_X_train, os_y_train = os.fit_sample(X, y.ravel())
	# os_X_train = pd.DataFrame(data=os_X_train, columns=X.columns)
	# os_y_train = pd.DataFrame(data=os_y_train, columns=['Total_time'])

	X = df.iloc[:,:-1]
	y = df.iloc[:,-1]


	# # convert categoral variables back to int
	# os_X_train['PQ'] = os_X_train['PQ'].astype(int)
	# os_X_train['K'] = os_X_train['K'].astype(int)

	# define preprocessor
	# preprocess = make_column_transformer(
	#     (OneHotEncoder(categories='auto'), ['PQ', 'K'])
	# )

	# define model
	# model = DecisionTreeClassifier()
	# model = SVR(kernel='rbf',C=1000.0, epsilon=0.01, gamma='auto')
	model = LinearRegression()

	scorer = make_scorer(mean_squared_error, greater_is_better=False)
	# scorer = make_scorer(explained_variance_score, greater_is_better=False)

	# define pipeline
	# over = SMOTE(sampling_strategy=0.1)
	# under = RandomUnderSampler(sampling_strategy=0.5)
	# steps = [('o', over), ('u', under), ('m', model)]
	# steps = [('p', preprocess),('m', model)]
	steps = [('m', model)]
	pipeline = Pipeline(steps=steps)
	
	# define evaluation procedure
	# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)

	# evaluate model
	scores = cross_val_score(pipeline, X, y,scoring=scorer, cv=cv, n_jobs=-1)
	# summarize performance
	print (scores)
	print('Score: %.3f' % scores.mean())
	#-0.439
def main(inputfilename):

	data = pd.read_csv(inputfilename)
	
	# if debug ==1:

	# 	print(data.columns)
	# 	print(data.info)

	if do_plots == 1:
		plot_graphs(data)

	if do_training == 1:
		main_logic2(data)

	



if __name__ == '__main__':
	if len(sys.argv) == 0 or len(sys.argv) < 2:
		print('ERROR: You have to pass dataset name')

	script = sys.argv[0]
	filename1 = sys.argv[1]

	

	main(filename1)