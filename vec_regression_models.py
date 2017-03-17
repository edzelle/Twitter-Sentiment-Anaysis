"""
Robert --- 2/27/2017. A few things I'm noticing.
(1) Our regression models are sometimes predicting extreme outliers (example: discounting score = 500). This is hurting the correlations.
"""

import csv, re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import pearsonr
from sklearn.model_selection import learning_curve
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

## ELI: this removes outliers from TRAINING data...want to experiment with using/not using this. ###
def remove_outliers(X, Y, threshold=2):
	m = np.mean(Y)
	sd = np.std(Y)
	mini = (m - threshold*sd)
	maxi = (m + threshold*sd)
	x_out, y_out = [], []
	for (x, y) in zip(X,Y):
		if (y>=mini) and (y<=maxi):
			x_out.append(x)
			y_out.append(y)
		else:
			print ('removed one')
	return (x_out, y_out)

### ELI: this is another function. At PREDICTION time, remove predictions that are outliers then report accuray ###
def remove_prediction_outliers(y_pred, y_test, threshold=1):  # thresdhold = # SD from the mean. 
	m = np.mean(y_pred)
	sd = np.std(y_pred)
	mini = (m - threshold*sd)
	maxi = (m + threshold*sd)
	print (m, sd, mini,maxi)
	pred_out = []
	actual_out = []
	for (pred, actual) in zip(y_pred, y_test):
		if (pred >= mini) & (pred <= maxi):
			pred_out.append(pred)
			actual_out.append(actual)
	return (pred_out, actual_out)
		
def plot_learning_curve(estimator, X, y, title = None, ylim = None, cv = None,
	n_jobs = 1, train_sizes = np.linspace(.1,1.0,10)):

	plt.figure()
	if title is not None:
		plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel('Training Examples')
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(
	estimator, X, y, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes)
	train_scores_mean = np.mean(train_scores, axis = 1)
	train_scores_std = np.std(train_scores, axis = 1)
	test_scores_mean = np.mean(test_scores, axis = 1)
	test_scores_std = np.std(test_scores, axis = 1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
					train_scores_mean + train_scores_std, alpha = 0.1,
					color = 'b')
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
					test_scores_mean + test_scores_std, alpha = 0.1, color = 'r')
	plt.plot(train_sizes, train_scores_mean, 'o-', color = 'b', label= "Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color = 'r',
			label = "Cross-validation score")
	plt.legend(loc= 'best')

	return plt



#### VARS #####
infile_name = "discounting_vectorized_normalized_constituency_parse_2_14_17_one_participant_per_line.csv"
score_col = 1
vector_col = 2

## Read in X, Y matrices from input file. 
### TODO. Some lines are being read in improperly, it seems we are reading '['']' as the x data. Let's look into why.
f = open(infile_name, 'r')
r = csv.reader(f)
csv.field_size_limit(999999999)
#next(r) # skip header.
X, Y = [], []
line_counter = 0 
for row in r:
	try: 
		
		line_counter += 1
		"""
		temp = []
		temp.append(float(row[0]))
		temp.append(float(row[1]))
		X.append(temp)
		Y.append(float(row[2]))
		"""
		# READ IN x, y data.
		y = float(row[score_col].lstrip(' ').rstrip(' '))		
		vec_string = row[vector_col].lstrip('[').rstrip(']')
		vec_list_of_strings = vec_string.split(', ')
		vec_floats = [float(s) for s in vec_list_of_strings if s != '']
		
		#ASSERT: x is not empty.
		#Skip lines if vectors are incomplete
		assert len(vec_floats) == 200

		# Append data to X, Y arrays.
		vec_floats = np.array(vec_floats)
		y = np.array(y)
		X.extend(vec_floats)
		Y.append(y)

		if (line_counter%25000 == 0):
			print(line_counter)

	except Exception as e:
		continue
		##print(e, vec_list_of_strings)
# convert X, Y to numpy arrays.
print("Converting to np arrays")
X = np.array(X)
Y = np.array(Y) 
print("Reshaping")
X = X.reshape(len(Y),200)

### ELI: this is new. Let's try using vs. not using this. Removes outliers from training data. ###
## REMOVE OUTLIERS FROM Y ##
(X, Y) = remove_outliers(X, Y, threshold = 1.5)  # threshold = # SD , we may need to tweak this. 



""" CROSS VALIDATION 'By Hand' """
mean_scores = {} # KEY = CLF; VALUE = M
for iter in range(50):
	
	# TRAIN TEST SPLIT #
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
	# TO DO: 
	# 1. Intercept
	# 2. Try different alpha values for Ridge and Lasso
	# 3. Plot train error vs training epoch


	for (model, name) in [(linear_model.LinearRegression(fit_intercept = True), 'Linear Regression'),
	(linear_model.Ridge(alpha = 1.2, fit_intercept = True), 'Ridge Regression'),
	(linear_model.Lasso(alpha = 1.2, fit_intercept = True), 'Lasso Regression'),
#	(DecisionTreeRegressor(presort = True), 'Decision Tree Regression'),
	(svm.SVR(kernel = 'linear', C = 1.0), 'Support Vector Machine Linear Regression')
#	(RandomForestRegressor(n_estimators=100, random_state = 0), 'Random Forest Regressor'),
#	(GradientBoostingRegressor(loss = 'ls', learning_rate = .02, n_estimators = 1500, alpha = .9), 'Gradient Boosting Regressor'),
#	(MLPRegressor(hidden_layer_sizes=(100,1), activation = 'logistic'), 'MLPRegressor')
	]:
	
		# FIT MODEL AND PREDICT Y TEST #
		clf = model
		clf.fit(X_train, Y_train)
		pred = clf.predict(X_test)
		
		## ELIMINATE OUTLIERS FROM PREDICTIONS ## .   ## #ELI you may want to play with this line ##
		(pred_no_outliers, actual_no_outliers) = remove_prediction_outliers(
			pred, Y_test, threshold = 1.5) # threshold = +/- how many SD to eliminate.
			
		# EVALUATE MODEL #
		r = pearsonr(pred, Y_test)[0]
		print ('MODEL = %s, r = %s'%(name, r))
		
		# PLOT PREDICTIONS VERSUS ACTUAL # 
		plt.scatter(pred, Y_test)
		plt.title(name)
		plt.xlabel('PREDICTED')
		plt.ylabel('ACTUAL')
		plt.show()


		# PLOT VALIDATION CURVE #
		plot_learning_curve(estimator = model, X = X, y = Y, title = name, cv = 5)
		plt.show()
		# UPDATE MEAN SCORE #
		if name in mean_scores.keys():
			old = mean_scores[name]
			old.append(r)
			mean_scores[name] = old
		else:
			mean_scores[name] = [r]
	

	# PRINT MEAN SCORES #
	print ('Mean scores \n\n')
	for k in mean_scores.keys():
		print ('CLF = %s, MEAN SCORE = %s'%(k, np.nanmean(mean_scores[k])))
