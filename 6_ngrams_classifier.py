"""
Progress as of Wednesday 3/8. 
Putting 1 user per line may be working, specifically with ridge regression. Ridge does the
best and trains the fastest. BUT, need more users.
What is new is I cleaned the data, delete RT, require 20+ tweets per user.
"""
import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neural_network import MLPRegressor
import time
	## Given 2 arrays, example X and Y data, drop N percent of the data, making sure that if we keep an X value we
# also keep the corresponding y value #
# returns :  (X_train, y_train, X_test, y_test)
def drop_X_percent(arr1,arr2, drop_percent, test_percent = 30):
	
	## define lists of INDICES to slice output arrays to ##
	X_train_indices = []
	y_train_indices = []
	X_test_indices = []
	y_test_indices = []

	idx = 0		
	for (x,y) in zip(arr1, arr2):
			
		# draw a random number
		r = random.randint(0,100)
		
			# only include if r >= percent
		if r >= test_percent:
			X_train_indices.append(idx)
			y_train_indices.append(idx)
		
		# else, try to include in the test set.
		elif random.randint(0,100) <= test_percent:
			X_test_indices.append(idx)
			y_test_indices.append(idx)
		
		# update index
		idx += 1
		
	# finally create output matrices by slicing.
	X_train = arr1[X_train_indices]
	y_train = arr2[y_train_indices]
	X_test = arr1[X_test_indices]
	y_test = arr2[y_test_indices]
		
	return (X_train, y_train, X_test, y_test)
	
	
		
## FUNCTION TO PLOT LEARNING CURVES ##
def plot_learning_curve(estimator, X, y, title = None, ylim = None, cv = None,
	n_jobs = 1, train_sizes = np.linspace(.1,1.0,10)):
	
	plt.figure()
	if title is not None:
		plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel('Training Examples')
	plt.ylabel("Score")
	train_scores, test_scores = [],[]
	for size in train_sizes:
		
		# use only train_size percent of data. Convert X and y to numpy arrays.
		(X_train, y_train, X_test, y_test) = drop_X_percent(X,y, drop_percent = 1-size)
		
		# fit and score #
		estimator.fit(X_train, y_train)
		y_test_pred = estimator.predict(X_test)
		(r_test,p) = pearsonr(y_test, y_test_pred)
		y_train_pred = estimator.predict(X_train)
		(r_train,p) = pearsonr(y_train, y_train_pred)
		
		# append to results.
		train_scores.append(r_train)
		test_scores.append(r_test)
		
	plt.grid()
	plt.plot(train_sizes, train_scores, 'o-', color = 'b', label= "Training score")
	plt.plot(train_sizes, test_scores, 'o-', color = 'r',
			label = "Test score")
	plt.legend(loc= 'best')
	return plt
	##VARS ##
n = 20
#infile = "5_discounting_ready_to_classify_%s_tweets_per_line_permuted_fake_data.csv"%n
infile = "5_discounting_ready_to_classify_%s_tweets_per_line.csv"%n

## READ DATA ##
print('reading data')
df = pd.read_csv(infile)
#df = df[df.ADJ_PUMPS < 50] # fit pumps <50
text = df.TEXT.values
y = df.RI.values
assert len(text) == len(y) # assert X and Y have same shape.
	## REPRESENT THE TEXT AS TF-IDF TRANSFORMED 1GRAM VECTORS ##
print ('creating ngram vectors and doing tf-idf transform')
tfidf = TfidfVectorizer(stop_words = 'english') # does ngram transform then tfidf.
X_tfidf = tfidf.fit_transform(text)


ridge_rs = []
for _ in range(2):
	
		## TRAIN TEST SPLIT ##
	X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size = 0.3)
		## TRAIN, TEST MODELS AND REPORT SCORE ##
	#models = [Ridge(), LinearRegression(), DecisionTreeRegressor()]
	#labels = ['Ridge Regression', 'Linear Regression', 'Decision Tree Regression']
	# Eli: can you please add the SVM?
	models = [Ridge(alpha = 1.0, fit_intercept = True, normalize = False),svm.SVR(kernel = 'linear', C = 1.0)]
	labels = ['Ridge Regression','Support Vector Machine Linear Regression' ]
	for (clf, label) in zip(models, labels):
		
		# close old plots # 
		plt.close('all')
		
		# fit and score #
		print('training %s'%label)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		(r,p) = pearsonr(y_test, y_pred)
		
		# scatterplot of TEST predicted, actual #
		plt.scatter(y_pred, y_test)
		plt.xlabel('PREDICTED')
		plt.ylabel('ACTUAL')
		plt.title('%s iteration # %s : r = %s, p = %s'%(label,_,r,p))
		plt.savefig('%s_%s.png'%(label, _))
		
		
		# LEARNING CURVE #
		plt_lc = plot_learning_curve(clf, X_tfidf, y, title = label)
		plt_lc.savefig('learning_curve_%s_iteration_%s.png'%(label, _))
		
		
		# add to average r for ridge #
		ridge_rs.append(r)
		
		# print iteration # and result.
		print('Iteration %s : r = %s'%(_, r))
	
print('Mean r for ridge: r = %s'%np.mean(r))
		
"""
QUARRY 
	## MANUAL TRAIN TEST SPLIT ###
N = len(y)
X_train = X_tfidf[0 : int(0.7*N)]
X_test = X_tfidf[int(0.7*N) : ]
y_train = y[0: int(0.7*N)]
y_test = y[int(0.7*N) : ]
# assert shapes are equal across X, Y.
assert X_train.shape[0] == len(y_train)
assert X_test.shape[0] == len(y_test)

"""
