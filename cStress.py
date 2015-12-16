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
from sklearn import svm, metrics, cross_validation, preprocessing
from sklearn.cross_validation import LabelKFold
from sklearn import grid_search
from pprint import pprint
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

args = parser.parse_args()


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


def f1Bias_scorer_CV(probs, y):

	precision, recall, thresholds = metrics.precision_recall_curve(y, probs)

	f1 = 0.0
	for i in range(0,len(thresholds)):
		if not(precision[i] == 0 and recall[i] == 0):
			f = 2 * (precision[i]*recall[i]) / (precision[i]+recall[i])
			if f > f1:
				f1 = f
				bias = thresholds[i]

	return f1,bias


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

	for i, (train, test) in enumerate(cv):
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

	#svmOutput('svmDataFile.txt',traindata, trainlabels)


	# fit the model

	clf = svm.SVC(C=0.70710678118654757, cache_size=2000,
				  class_weight={0: 0.20000000000000001, 1: 0.80000000000000004},
				  coef0=0.0,decision_function_shape=None, degree=3, gamma=2.8284271247461903,
				  kernel='rbf', max_iter=-1, probability=True, random_state=None,
				  shrinking=True, tol=0.001, verbose=False)

	'''
    clf = svm.SVC(C=0.53000000000000003, cache_size=200,
				 class_weight={0: 0.28000000000000003, 1: 0.71999999999999997}, coef0=0.0,
				  decision_function_shape=None, degree=3, gamma=0.75, kernel='rbf',
				  max_iter=-1, probability=True, random_state=None, shrinking=True,
				  tol=0.001, verbose=False)
	'''
	normalizer = preprocessing.StandardScaler()
	traindata = normalizer.fit_transform(traindata)


	lkf = LabelKFold(subjects, n_folds=len(np.unique(subjects)))
	CV_probs = cross_val_probs(clf,traindata,trainlabels,lkf)

	f1,bias = f1Bias_scorer_CV(CV_probs, trainlabels)
	print f1,bias
	predicted = np.asarray(CV_probs >= bias,dtype=np.int)


	saveModel('model.txt',clf,normalizer,bias)

	# Write ROC code here for Bias search and replace following code
#	predicted = cross_validation.cross_val_predict(clf, traindata, trainlabels, cv=lkf)
	print("Cross-Subject (" + str(len(np.unique(subjects))) + "-fold) Validation Prediction")
	print("Accuracy: " + str(metrics.accuracy_score(trainlabels, predicted)))
	print(metrics.classification_report(trainlabels, predicted))
	print(metrics.confusion_matrix(trainlabels, predicted))
	print("Subjects: " + str(np.unique(subjects)))

	print("Best learned parameters")

	delta = 0.1
	parameters = {'kernel': ['linear','rbf'],
				  'C': [2 ** x for x in np.arange(-12, 12, 0.5)],
				  'gamma': [2 ** x for x in np.arange(-12, 12, 0.5)],
				  'class_weight': [{0: w, 1: 1 - w} for w in np.arange(0.0, 1.0, delta)]}
	svr = svm.SVC(probability=True, verbose=False, cache_size=2000)

	# clf = grid_search.GridSearchCV(svr, parameters, cv=lkf, n_jobs=-1, scoring='f1', verbose=1, iid=False)
	clf = grid_search.RandomizedSearchCV(svr, parameters, cv=lkf, n_jobs=-1, scoring=f1Bias_scorer, n_iter=100,
										 verbose=1, iid=False)
	# write custom scoring function for ROC-bias optimization

	clf.fit(traindata, trainlabels)


	CV_probs = cross_val_probs(clf.best_estimator_,traindata,trainlabels,lkf)

	f1,bias = f1Bias_scorer_CV(CV_probs, trainlabels)
	print f1,bias
	saveModel('model.txt',clf.best_estimator_,normalizer,bias)


	predicted = np.asarray(CV_probs >= bias,dtype=np.int)
	print("Cross-Subject (" + str(len(np.unique(subjects))) + "-fold) Validation Prediction")
	print("Accuracy: " + str(metrics.accuracy_score(trainlabels, predicted)))
	print(metrics.classification_report(trainlabels, predicted))
	print(metrics.confusion_matrix(trainlabels, predicted))
	print("Subjects: " + str(np.unique(subjects)))

	resultProbabilities = CV_probs

	# pprint(clf.get_params())
	pprint(clf.best_estimator_)


	pprint(clf.grid_scores_)
