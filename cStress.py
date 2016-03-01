# Copyright (c) 2015, University of Memphis, MD2K Center of Excellence
#  - Timothy Hnat <twhnat@memphis.edu>
#  - Karen Hovsepian <karoaper@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from collections import Counter
from pathlib import Path
import math
import numpy as np
import scipy
from pprint import pprint
import time
from collections import Mapping, namedtuple, Sized

from sklearn import svm, metrics, cross_validation, preprocessing
from sklearn.cross_validation import LabelKFold
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV,ParameterSampler,ParameterGrid
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone,is_classifier
from sklearn.utils.validation import _num_samples, indexable
from sklearn.cross_validation import check_cv

import json
from json import JSONEncoder


# Command line parameter configuration

parser = argparse.ArgumentParser(description='Train and evaluate the cStress model')
parser.add_argument('--featureFolder', dest='featureFolder', required=True,
					help='Directory containing feature files')
parser.add_argument('--groundtruthFile', dest='groundtruthFile', required=True,
					help='CSV file with groundtruth')
parser.add_argument('--parameterFile', dest='parameterFile', required=True, help='Model configuration parameters')
parser.add_argument('--featureStart', type=int, required=False, dest='featureStart',
					help='Specify which feature in the files to start with')
parser.add_argument('--featureEnd', type=int, required=False, dest='featureEnd',
					help='Specify which feature in the files to endwith')
parser.add_argument('--scorer', type=str, required=True, dest='scorer',
					help='Specify which scorer function to use')
parser.add_argument('--whichsearch', type=str, required=True, dest='whichsearch',
					help='Specify which search function to use (GridSearch or RandomizedSearch')
parser.add_argument('--n_iter', type=int, required=False, dest='n_iter',
					help='If Randomized Search is used, how many iterations to use')

args = parser.parse_args()



def cv_fit_and_score(estimator, X, y, scorer, parameters, cv,):
	"""Fit estimator and compute scores for a given dataset split.
	Parameters
	----------
	estimator : estimator object implementing 'fit'
		The object to use to fit the data.
	X : array-like of shape at least 2D
		The data to fit.
	y : array-like, optional, default: None
		The target variable to try to predict in the case of
		supervised learning.
	scorer : callable
		A scorer callable object / function with signature
		``scorer(estimator, X, y)``.
	parameters : dict or None
		Parameters to be set on the estimator.
	cv:	Cross-validation fold indeces
	Returns
	-------
	score : float
		CV score on whole set.
	parameters : dict or None, optional
		The parameters that have been evaluated.
	"""
	estimator.set_params(**parameters)
	cv_probs_ = cross_val_probs(estimator,X,y,cv)
	score = scorer(cv_probs_,y)

	return [score,parameters]#scoring_time]



class ModifiedGridSearchCV(GridSearchCV):
	def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
				 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
				 pre_dispatch='2*n_jobs', error_score='raise'):

		super(ModifiedGridSearchCV, self).__init__(
			estimator, param_grid,scoring, fit_params, n_jobs, iid,
			refit, cv, verbose, pre_dispatch, error_score)


 	def fit(self, X, y):
 		"""Actual fitting,  performing the search over parameters."""

 		parameter_iterable = ParameterGrid(self.param_grid)

		estimator = self.estimator
		cv = self.cv

		n_samples = _num_samples(X)
		X, y = indexable(X, y)

		if y is not None:
			if len(y) != n_samples:
				raise ValueError('Target variable (y) has a different number '
								 'of samples (%i) than data (X: %i samples)'
								 % (len(y), n_samples))
		cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

		if self.verbose > 0:
			if isinstance(parameter_iterable, Sized):
				n_candidates = len(parameter_iterable)
				print("Fitting {0} folds for each of {1} candidates, totalling"
					  " {2} fits".format(len(cv), n_candidates,
										 n_candidates * len(cv)))

		base_estimator = clone(self.estimator)

		pre_dispatch = self.pre_dispatch

		out = Parallel(
			n_jobs=self.n_jobs, verbose=self.verbose,
			pre_dispatch=pre_dispatch
		)(
			delayed(cv_fit_and_score)(clone(base_estimator), X, y, self.scoring,
									parameters,cv=cv)
				for parameters in parameter_iterable)

		best = sorted(out,reverse=True)[0]
		self.best_params_ = best[1]
		self.best_score_ = best[0]

		if self.refit:
			# fit the best estimator using the entire dataset
			# clone first to work around broken estimators
			best_estimator = clone(base_estimator).set_params(
				**best[1])
			if y is not None:
				best_estimator.fit(X, y, **self.fit_params)
			else:
				best_estimator.fit(X, **self.fit_params)
			self.best_estimator_ = best_estimator

		return self


class ModifiedRandomizedSearchCV(RandomizedSearchCV):
	def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
				 fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
				 verbose=0, pre_dispatch='2*n_jobs', random_state=None,
				 error_score='raise'):

		super(ModifiedRandomizedSearchCV, self).__init__(estimator=estimator, param_distributions=param_distributions,
														 n_iter=n_iter, scoring=scoring,random_state = random_state,
														 fit_params=fit_params,n_jobs=n_jobs, iid=iid, refit=refit,
														 cv=cv, verbose=verbose,pre_dispatch=pre_dispatch, error_score=error_score)


	def fit(self, X, y):
		"""Actual fitting,  performing the search over parameters."""

		parameter_iterable = ParameterSampler(self.param_distributions,
										  self.n_iter,
										  random_state=self.random_state)
		estimator = self.estimator
		cv = self.cv

		n_samples = _num_samples(X)
		X, y = indexable(X, y)

		if y is not None:
			if len(y) != n_samples:
				raise ValueError('Target variable (y) has a different number '
								 'of samples (%i) than data (X: %i samples)'
								 % (len(y), n_samples))
		cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

		if self.verbose > 0:
			if isinstance(parameter_iterable, Sized):
				n_candidates = len(parameter_iterable)
				print("Fitting {0} folds for each of {1} candidates, totalling"
					  " {2} fits".format(len(cv), n_candidates,
										 n_candidates * len(cv)))

		base_estimator = clone(self.estimator)

		pre_dispatch = self.pre_dispatch

		out = Parallel(
			n_jobs=self.n_jobs, verbose=self.verbose,
			pre_dispatch=pre_dispatch
		)(
			delayed(cv_fit_and_score)(clone(base_estimator), X, y, self.scoring,
									parameters,cv=cv)
				for parameters in parameter_iterable)

		best = sorted(out,reverse=True)[0]
		self.best_params_ = best[1]
		self.best_score_ = best[0]

		if self.refit:
			# fit the best estimator using the entire dataset
			# clone first to work around broken estimators
			best_estimator = clone(base_estimator).set_params(
				**best[1])
			if y is not None:
				best_estimator.fit(X, y, **self.fit_params)
			else:
				best_estimator.fit(X, **self.fit_params)
			self.best_estimator_ = best_estimator

		return self



def decodeLabel(label):
	label = label[:2]  # Only the first 2 characters designate the label code

	mapping = {'c1': 0, 'c2': 1, 'c3': 1, 'c4': 0, 'c5': 0, 'c6': 0, 'c7': 2, }

	return mapping[label]


def readFeatures(folder):
	features = []

	path = Path(folder)
	files = list(path.glob('**/org.md2k.cstress.fv.csv'))

	for f in files:
		participantID = int(f.parent.name[2:])
		with f.open() as file:
			for line in file.readlines():
				parts = [x.strip() for x in line.split(',')]

				featureVector = [participantID, int(parts[0])]
				featureVector.extend([float(p) for p in parts[1:]])

				features.append(featureVector)

	return features


def readStressmarks(folder):
	features = []

	path = Path(folder)
	files = list(path.glob('**/stress_marks.txt'))

	for f in files:
		participantID = int(f.parent.name[2:])

		with f.open() as file:
			for line in file.readlines():
				parts = [x.strip() for x in line.split(',')]
				label = parts[0][:2]
				features.append([participantID, label, int(parts[2]), int(parts[3])])

	return features


def readGroundtruth(file):
	features = []

	with open(file) as file:
		for line in file.readlines():
			parts = [int(x.strip()) for x in line.split(',')]

			features.append(parts)
	return features


def checkStressMark(stressMark, pid, starttime):
	endtime = starttime + 60000  # One minute windows
	result = []
	for line in stressMark:
		[id, gt, st, et] = line

		if id == pid and (gt not in ['c7']):
			if (starttime > st) and (endtime < et):
				result.append(gt)

	data = Counter(result)
	return data.most_common(1)


def analyze_events_with_features(features, stress_marks):
	featureLabels = []
	finalFeatures = []
	subjects = []

	startTimes = {}
	for pid, label, start, end in stress_marks:
		if label == 'c4':
			if pid not in startTimes:
				startTimes[pid] = np.inf

			startTimes[pid] = min(startTimes[pid], start)

	for line in features:
		id = line[0]
		ts = line[1]
		f = line[2:]

		if ts < startTimes[id]:
			continue  # Outside of starting time

		label = checkStressMark(stress_marks, id, ts)
		if len(label) > 0:
			stressClass = decodeLabel(label[0][0])

			featureLabels.append(stressClass)
			finalFeatures.append(f)
			subjects.append(id)

	return finalFeatures, featureLabels, subjects


def get_svmdataset(traindata, trainlabels):
	input = []
	output = []
	foldinds = []

	for i in range(len(trainlabels)):
		if trainlabels[i] == 1:
			foldinds.append(i)

		if trainlabels[i] == 0:
			foldinds.append(i)

	input = np.array(input, dtype='float64')
	return output, input, foldinds


def reduceData(data, r):
	result = []
	for d in data:
		result.append([d[i] for i in r])
	return result


def f1Bias_scorer(estimator, X, y,ret_bias=False):

	probas_ = estimator.predict_proba(X)
	precision, recall, thresholds = metrics.precision_recall_curve(y, probas_[:,1])

	f1 = 0.0
	for i in range(0,len(thresholds)):
		if not(precision[i] == 0 and recall[i] == 0):
			f = 2 * (precision[i]*recall[i]) / (precision[i]+recall[i])
			if f > f1:
				f1 = f
				bias = thresholds[i]

	if ret_bias:
		return f1,bias
	else:
		return f1


def Twobias_scorer_CV(probs, y,ret_bias=False):


	db = np.transpose(np.vstack([probs,y]))
	db = db[np.argsort(db[:,0]),:]

	pos = np.sum(y == 1)
	n = len(y)
	neg = n-pos
	tp,tn = pos , 0
	lost = 0

	optbias = []
	minloss = 1

	for i in range(n):
#		p = db[i,1]
		if db[i,1]==1:				 #positive
			tp-=1.0
		else:
			tn+=1.0

#		v1 = tp/pos
#		v2 = tn/neg
		if tp/pos >= 0.95 and tn/neg >= 0.95:
			optbias = [db[i,0],db[i,0]]
			continue

		running_pos = pos
		running_neg = neg
		running_tp = tp
		running_tn = tn

		for j in range(i+1,n):
#			p1 = db[j,1]
			if db[j,1]==1:				 #positive
				running_tp -= 1.0
				running_pos -= 1
			else:
				running_neg -= 1

			lost = (j-i)*1.0/n
			if running_pos == 0 or running_neg == 0:
				break

#			v1 = running_tp/running_pos
#			v2 = running_tn/running_neg

			if running_tp/running_pos >= 0.95 and running_tn/running_neg >= 0.95 and lost < minloss:
				minloss = lost
				optbias = [db[i,0],db[j,0]]


	if ret_bias:
		return -minloss,optbias
	else:
		return -minloss




def f1Bias_scorer_CV(probs, y,ret_bias=False):

	precision, recall, thresholds = metrics.precision_recall_curve(y, probs)

	f1 = 0.0
	for i in range(0,len(thresholds)):
		if not(precision[i] == 0 and recall[i] == 0):
			f = 2 * (precision[i]*recall[i]) / (precision[i]+recall[i])
			if f > f1:
				f1 = f
				bias = thresholds[i]

	if ret_bias:
		return f1,bias
	else:
		return f1


def svmOutput(filename, traindata, trainlabels):

	with open(filename, 'w') as f:
		for i in range(0, len(trainlabels)):
			f.write(str(trainlabels[i]))
			for fi in range(0, len(traindata[i])):
				f.write(" " + str(fi+1) + ":" + str(traindata[i][fi]))

			f.write("\n")


def saveModel(filename,model,normparams,bias=0.5):


	class Object:
		def to_JSON(self):
			return json.dumps(self, default=lambda o: o.__dict__,
				sort_keys=True, indent=4)

	class Kernel(Object):
		def __init__(self,type,parameters):
			self.type = type
			self.parameters = parameters

	class KernelParam(Object):
		def __init__(self,name,value):
			self.name = name;
			self.value = value

	class Support(Object):
		def __init__(self,dualCoef,supportVector):
			self.dualCoef = dualCoef
			self.supportVector = supportVector

	class NormParam(Object):
		def __init__(self,mean,std):
			self.mean = mean
			self.std = std

	class SVCModel(Object):
		def __init__(self,modelName,modelType,intercept,bias,probA,probB,kernel,support,normparams):
			self.modelName = modelName;
			self.modelType = modelType;
			self.intercept = intercept;
			self.bias = bias;
			self.probA = probA;
			self.probB = probB;
			self.kernel = kernel
			self.support = support
			self.normparams = normparams


	model = SVCModel('cStress','svc',model.intercept_[0],bias,model.probA_[0],model.probB_[0],Kernel('rbf',[KernelParam('gamma',model._gamma)]),[Support(model.dual_coef_[0][i],list(model.support_vectors_[i])) for i in range(len(model.dual_coef_[0]))],[NormParam(normparams.mean_[i],normparams.scale_[i]) for i in range(len(normparams.scale_))])

	with open(filename,'w') as f:
		print >> f, model.to_JSON()




def cross_val_probs(estimator,X,y,cv):
	probs = np.zeros(len(y))

	for train, test in cv:
		temp = estimator.fit(X[train], y[train]).predict_proba(X[test])
		probs[test] = temp[:,1]

	return probs



# This tool accepts the data produced by the Java cStress implementation and trains and evaluates an SVM model with
# cross-subject validation
if __name__ == '__main__':
	featuresused = range(args.featureStart - 1, args.featureEnd)

	features = readFeatures(args.featureFolder)
	groundtruth = readStressmarks(args.featureFolder)

	traindata, trainlabels, subjects = analyze_events_with_features(features, groundtruth)
	traindata = reduceData(traindata, featuresused)

	traindata = np.asarray(traindata,dtype=np.float64)
	trainlabels = np.asarray(trainlabels)



	normalizer = preprocessing.StandardScaler()
	traindata = normalizer.fit_transform(traindata)

	lkf = LabelKFold(subjects, n_folds=len(np.unique(subjects)))

	print("Best learned parameters")

	delta = 0.1
	parameters = {'kernel': ['rbf'],
				  'C': [2 ** x for x in np.arange(-12, 12, 0.5)],
				  'gamma': [2 ** x for x in np.arange(-12, 12, 0.5)],
				  'class_weight': [{0: w, 1: 1 - w} for w in np.arange(0.0, 1.0, delta)]}
	# parameters = {'kernel': ['rbf'],
	# 			  'C': [2 ** x for x in np.arange(5, 6, 1)],
	# 			  'gamma': [2 ** x for x in np.arange(3, 5, 1)],
	# 			  'class_weight': [{0: w, 1: 1 - w} for w in np.arange(0.0, 1.0, delta)]}


	svc = svm.SVC(probability=True, verbose=False, cache_size=2000)


	if args.scorer == 'f1':
		scorer = f1Bias_scorer_CV
	else:
		scorer = Twobias_scorer_CV


	if args.whichsearch == 'grid':
		clf = ModifiedGridSearchCV(svc, parameters, cv=lkf, n_jobs=-1, scoring=scorer, verbose=1, iid=False)
	else:
		clf = ModifiedRandomizedSearchCV(estimator=svc, param_distributions=parameters, cv=lkf, n_jobs=-1, scoring=scorer, n_iter=args.n_iter,
										 verbose=1, iid=False)

	clf.fit(traindata, trainlabels)
	pprint(clf.best_params_)


	CV_probs = cross_val_probs(clf.best_estimator_,traindata,trainlabels,lkf)
	score,bias = scorer(CV_probs, trainlabels,True)
	print score,bias
	if not bias == []:
		saveModel('model.txt',clf.best_estimator_,normalizer,bias)

		n = len(trainlabels)

		if args.scorer == 'f1':
			predicted = np.asarray(CV_probs >= bias,dtype=np.int)
			classified = range(n)
		else:
			classified = np.where(np.logical_or(CV_probs <= bias[0],CV_probs >= bias[1]))[0]
			predicted = np.asarray(CV_probs[classified]>=bias[1],dtype=np.int)

		print("Cross-Subject (" + str(len(np.unique(subjects))) + "-fold) Validation Prediction")
		print("Accuracy: " + str(metrics.accuracy_score(trainlabels[classified], predicted)))
		print(metrics.classification_report(trainlabels[classified], predicted))
		print(metrics.confusion_matrix(trainlabels[classified], predicted))
		print("Lost: %d (%f%%)" % (n - len(classified),(n - len(classified))*1.0/n))
		print("Subjects: " + str(np.unique(subjects)))
	else:
		print "Results not good"
