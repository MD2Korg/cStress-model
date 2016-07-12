import argparse
import json
from collections import Sized
from pprint import pprint

import numpy as np
from pathlib import Path
from sklearn import svm, metrics, preprocessing
from sklearn.base import clone, is_classifier
from sklearn.cross_validation import LabelKFold
from sklearn.cross_validation import check_cv
from sklearn.externals.joblib import Parallel, delayed
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV, ParameterSampler, ParameterGrid
from sklearn.utils.validation import _num_samples, indexable

def cross_val_probs(estimator, X, y, cv):
    probs = np.zeros(len(y))

    for train, test in cv:
        temp = estimator.fit(X[train], y[train]).predict_proba(X[test])
        probs[test] = temp[:, 1]

    return probs


def saveModel(filename, model, normparams, bias=0.5):
    class Object:
        def to_JSON(self):
            return json.dumps(self, default=lambda o: o.__dict__,
                              sort_keys=True, indent=4)

    class Kernel(Object):
        def __init__(self, type, parameters):
            self.type = type
            self.parameters = parameters

    class KernelParam(Object):
        def __init__(self, name, value):
            self.name = name;
            self.value = value

    class Support(Object):
        def __init__(self, dualCoef, supportVector):
            self.dualCoef = dualCoef
            self.supportVector = supportVector

    class NormParam(Object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

    class SVCModel(Object):
        def __init__(self, modelName, modelType, intercept, bias, probA, probB, kernel, support, normparams):
            self.modelName = modelName;
            self.modelType = modelType;
            self.intercept = intercept;
            self.bias = bias;
            self.probA = probA;
            self.probB = probB;
            self.kernel = kernel
            self.support = support
            self.normparams = normparams

    model = SVCModel('cStress', 'svc', model.intercept_[0], bias, model.probA_[0], model.probB_[0],
                     Kernel('rbf', [KernelParam('gamma', model._gamma)]),
                     [Support(model.dual_coef_[0][i], list(model.support_vectors_[i])) for i in
                      range(len(model.dual_coef_[0]))],
                     [NormParam(normparams.mean_[i], normparams.scale_[i]) for i in range(len(normparams.scale_))])

    with open(filename, 'w') as f:
        print >> f, model.to_JSON()


# This tool accepts the data produced by the Java cStress implementation and trains and evaluates an SVM model with
# cross-subject validation
if __name__ == '__main__':
    # features = readFeatures(args.featureFolder, args.featureFile)
    import csv
    f = open("featureFile_new.csv")
    inputdata = csv.reader(f)

    features = []
    groundtruth = []

    for r in inputdata:
        features.append(map(float, r[:-1]))
        groundtruth.append(int(r[-1]))


    traindata = np.asarray(features, dtype=np.float64)
    trainlabels = np.asarray(groundtruth)

    normalizer = preprocessing.StandardScaler()
    traindata = normalizer.fit_transform(traindata)

    delta = 0.1

    parameters = {'kernel': ['rbf'],
                  'C': [2 ** x for x in np.arange(-12, 12, 0.5)],
                  'gamma': [2 ** x for x in np.arange(-12, 12, 0.5)],
                  'class_weight': [{0: w, 1: 1 - w} for w in np.arange(0.0, 1.0, delta)]}

    svc = svm.SVC(probability=True, verbose=False, cache_size=2000)


    from sklearn.cross_validation import KFold

    lkf = KFold(len(groundtruth), 2)

    # clf = GridSearchCV(svc, parameters, cv=lkf, n_jobs=-1, scoring='roc_auc', verbose=1, iid=False)
    clf = RandomizedSearchCV(svc, parameters, cv=lkf, n_iter=2, n_jobs=-1, scoring='roc_auc', verbose=1, iid=False)


    clf.fit(traindata, trainlabels)
    pprint(clf.best_params_)

    CV_probs = cross_val_probs(clf.best_estimator_, traindata, trainlabels, lkf)

    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(trainlabels, CV_probs, pos_label=1)
    # print(fpr)
    # print(tpr)
    # print(thresholds)

    output = []
    for i in xrange(0,len(fpr)):
        output.append( [i, fpr[i], tpr[i], thresholds[i]] )

    pprint(output)


    saveModel('puffmarker_model_new.json', clf.best_estimator_, normalizer)