#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### [will be completed later in the code]
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# First feature list for feature selection, the choosing for this first set of
# features is described in the answers to questions write-up.
features_list_try = ['poi','salary','total_stock_value','total_payments',
                      'restricted_stock','exercised_stock_options', 'expenses',
                      'to_messages', 'from_messages', 'from_poi_to_this_person',
                      'from_this_person_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop( "TOTAL", 0 ) # REMOVE "TOTAL" OUTLIER (as considered in lesson 7)


# *******************************************************************
# EXPLORE DATASET
print "EXPLORE DATASET"
# Total number of data points
print "Number of people in dataset:", len(data_dict)
# POIs (Persons Of Interest) in the dataset
poi_total = 0
for features in data_dict.values():
    if features["poi"] == True:
        poi_total += 1
print "Number of POIs:", poi_total
# Features for each person
print "Number of features for each person:", len(data_dict.values()[0])

# Explore NaN occurrences on each feature

nan_count = [0] * len(data_dict.values()[0])
for features in data_dict.values():
    current_index = 0
    for f in features:
        if features[f] == 'NaN':
            nan_count[current_index] += 1
        current_index += 1

percentages = [float(e) / len(data_dict) for e in nan_count]
nan_percentage_dict = {}
index = 0
for key, value in data_dict.values()[0].iteritems():
    nan_percentage_dict[key] = percentages[index]
    index += 1

print "NaN proportion for each variable:\n", nan_percentage_dict
print

# *******************************************************************

# 'try' for first exploration on feature selection
data_try = featureFormat(data_dict, features_list_try, sort_keys = True)
labels_try, features_try = targetFeatureSplit(data_try)

# First I will explore feature importances with a Decision Tree as a starting point
from sklearn.cross_validation import train_test_split
f_train_1, f_test_1, l_train_1, l_test_1 = train_test_split(features_try, labels_try)

from sklearn.tree import DecisionTreeClassifier
clf_1 = DecisionTreeClassifier()
clf_1 = clf_1.fit(f_train_1, l_train_1)

print "FEATURE SELECTION\n"
print "Feature importances with simple Decision Tree"
for index,importance in enumerate(clf_1.feature_importances_):
    print "Feature:", features_list_try[index+1]
    print "Importance:", importance
print

# After running this code several times, I suspect 'total_stock_value' and
# 'exercised_stock_options' may be correlated; since, when the importance of
# one of this features is high, the other one is low, and vice versa.
# The one that seems to be more consistently important is 'exercised_stock_options'.
# Also, in the email features one of 'from_poi_to_this_person' and 'from_this_person_to_poi'
# may be important. Other variables that seem consistently important are
# 'expenses' and 'restricted_stock'.


# *******************************************************************

### Task 3: Create new feature(s)

# As discussed in Lesson 11, I will create two new features, 'fraction_from_poi'
# and 'fraction_to_poi' so that, in a way, I will be scaling
# 'from_poi_to_this_person' and 'from_this_person_to_poi', so that these values
# can become more meaningful
def computeFraction( poi_messages, all_messages ):
    if poi_messages == "NaN" or all_messages == "NaN":
        fraction = 0.
    else:
        fraction = float(poi_messages)/float(all_messages)
    return fraction

for name in data_dict:
    fraction_from_poi = computeFraction(data_dict[name]["from_poi_to_this_person"],
                                       data_dict[name]["to_messages"])
    data_dict[name]["fraction_from_poi"] = fraction_from_poi
    fraction_to_poi = computeFraction(data_dict[name]["from_this_person_to_poi"],
                                       data_dict[name]["from_messages"])
    data_dict[name]["fraction_to_poi"] = fraction_to_poi
# *******************************************************************

# Since the dataset is not particularly large, and the labeling is skewed toward
# a large proportion of non-POIs, I figure I will keep the number of features low,
# at maximum of 4 features.
# However I will explicitly use feature selection tools from sklearn to confirm
# my rationale and test the new features:

# First SelectKBest: since this is a univariate feature test/selection method,
# no scaling needed.
# Ref: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
# Ref: http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
# Ref: http://stackoverflow.com/questions/25792012/feature-selection-using-scikit-learn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# 'try' for second exploration on feature selection, with new features
# Here SelectKBest will pick 4 features from the 5 features that were picked
# from the previous analysis.
features_list_try_2 = ['poi','exercised_stock_options', 'expenses',
                        'fraction_from_poi', 'fraction_to_poi', 'restricted_stock']
data_try_2 = featureFormat(data_dict, features_list_try_2, sort_keys = True)
labels_try_2, features_try_2 = targetFeatureSplit(data_try_2)
selector = SelectKBest(f_classif, k=4)
features_try_2_selected = selector.fit_transform(features_try_2, labels_try_2)
# Ref: http://stackoverflow.com/questions/21471513/sklearn-selectkbest-which-variables-were-chosen
features_selected_indices = selector.get_support(indices=True) + 1 # Since I will retrive them from
                                                                   # 'features_list_test_2', which
                                                                   # contains 'poi' as first entry
print "Features selected by 'SelectKBest':\n", features_list_try_2[features_selected_indices[0]]
print features_list_try_2[features_selected_indices[1]]
print features_list_try_2[features_selected_indices[2]]
print features_list_try_2[features_selected_indices[3]]
print


# *******************************************************************
# Now I will explore Principal Component Analysis with a set of features that take
# into account all of the features in the first set, plus the new created features.
# The principal components will not be any of the original features, but a linear
# combination of them.
# I still want to see if I can gain any further insight with the results.
# This is only an exploration, I will not use PCA in the final implementation.

# PCA needs scaling.

# Scaling of features
# Ref: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
from sklearn.preprocessing import MinMaxScaler
features_list_try_3 = ['poi','salary','total_stock_value','total_payments',
                        'restricted_stock','exercised_stock_options', 'expenses',
                        'to_messages', 'from_messages', 'from_poi_to_this_person',
                        'from_this_person_to_poi', 'fraction_from_poi', 'fraction_to_poi'] # Also used in Pipeline later on
data_try_3 = featureFormat(data_dict, features_list_try_3, sort_keys = True)
labels_try_3, features_try_3 = targetFeatureSplit(data_try_3)
scaler = MinMaxScaler()
features_try_3_scaled = scaler.fit_transform(features_try_3)
# PCA
# Ref: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# Ref:
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
features_try_3_selected = pca.fit(features_try_3_scaled)
print "PCA Exploration --"
print "PCA variances:", (pca.explained_variance_) # Check variances
print

# Ref: http://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ration-in-pca-with-sklearn
# With PCA "Each principal component is a linear combination of the original variables"
import numpy as np
import pandas as pd
i = np.identity(12)
coef = pca.transform(i)
# Ref: http://stackoverflow.com/questions/2142453/getting-list-without-kth-element-efficiently-and-non-destructively
only_features_3 = features_list_try_3[1:]
pc_features_components = pd.DataFrame(coef, columns=['PC(1)', 'PC(2)', 'PC(3)', 'PC(4)'],
                                      index=only_features_3)

print "PCA features components:\n", pc_features_components
print


# 'exercised_stock_options' is important in the second and third PC, but in neither
# case it is the main contributor. 'expenses' is the main contributor in PC(4)
# and very important in PC(3) along with 'fraction_to_poi', this last feature it's
# also very important in PC(2) with 'fraction_from_poi'. But in general I don't feel
# that these results give me a decisive insight, so I will keep with the features
# chosen by SelectKBest in the previous process.
# *******************************************************************

### Task 1: Select what features you'll use. [COMPLETION]
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','exercised_stock_options', 'expenses', 'fraction_from_poi','restricted_stock']
### Store to my_dataset for easy export below.
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# *******************************************************************

### Task 1: Select what features you'll use. -----------------------[REVIEW]
# After the first review I was asked to implement a more rigorous approach for
# optimizing feature selection in an iterative process. So I will now start with
# all possible features and test decreasing sets of them according to their
# importances. I will then compare this results to the choice I made before.
print "*************************************************************"
features_review = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',
                   'loan_advances', 'bonus', 'restricted_stock_deferred', 'total_stock_value',
                   'shared_receipt_with_poi', 'long_term_incentive', 'exercised_stock_options',
                   'from_messages', 'other', 'from_poi_to_this_person', 'from_this_person_to_poi',
                   'deferred_income', 'expenses', 'restricted_stock', 'director_fees',
                   'fraction_from_poi', 'fraction_to_poi']


data_review = featureFormat(data_dict, features_review, sort_keys = True)
labels_r, features_r= targetFeatureSplit(data_review)

clf_review = DecisionTreeClassifier()
clf_review = clf_review.fit(features_r, labels_r)

print "**REVIEW FEATURE SELECTION** - Feature importances with Decision Tree - all features"
features_review_1 = []
for index,importance in enumerate(clf_review.feature_importances_):
    print "Feature:", features_review[index+1]
    print "Importance:", importance
    if importance > 0:
        features_review_1.append(features_review[index+1])
print

# First feature choosing: importance at least  larger than zero:
print "Features with importance larger than zero - first iteration:", features_review_1
print

# Second iteration:
data_review_2 = featureFormat(data_dict, features_review_1, sort_keys = True)
labels_r_2, features_r_2= targetFeatureSplit(data_review_2)
clf_review = clf_review.fit(features_r_2, labels_r_2)
print "**REVIEW FEATURE SELECTION** - Feature importances with Decision Tree - second iteration"
features_review_2 = []
for index,importance in enumerate(clf_review.feature_importances_):
    print "Feature:", features_review_1[index]
    print "Importance:", importance
    if importance > 0.1:
        features_review_2.append(features_review_1[index])
print
print "Features with importance larger than 0.1 - second iteration:", features_review_2
print

# Third iteration:
data_review_3 = featureFormat(data_dict, features_review_2, sort_keys = True)
labels_r_3, features_r_3= targetFeatureSplit(data_review_3)
clf_review = clf_review.fit(features_r_3, labels_r_3)
print "**REVIEW FEATURE SELECTION** - Feature importances with Decision Tree - third iteration"
features_review_3 = []
for index,importance in enumerate(clf_review.feature_importances_):
    print "Feature:", features_review_2[index]
    print "Importance:", importance
    if importance > 0.1:
        features_review_3.append(features_review_2[index])
print
print "Features - third iteration:", features_review_3
print "Final list"
print "*************************************************************"

# Update feature list
"""
features_list = ["poi"] + features_review_3
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
"""
# Recall and Precision using iterative feature selection:
#Final recall: 0.166666666667
#Final precision: 0.25

# Since this feature list gives worse results, I will try the previous set:
"""
features_list = ["poi"] + features_review_2
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
"""
# Recall and Precision using iterative feature selection:
#Final recall: 0.333333333333
#Final precision: 0.333333333333

# It is still a worse result, SO I WILL KEEP MY INITIAL FEATURE SELECTION

# *******************************************************************
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# *******************************************************************
# I will try a variety of classifiers training on the whole dataset

## Provided to give you a starting point. Try a variety of classifiers.
"""
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
"""

# I will record performance of this starting point -OUTPUT FROM tester.py (code provided)-
#
#GaussianNB()
#	Accuracy: 0.85607	Precision: 0.49347	Recall: 0.28350	F1: 0.36011	F2: 0.30987
#	Total predictions: 14000	True positives:  567	False positives:  582	False negatives: 1433	True negatives: 11418
#
# I notice a large number of predictions (14000) compared to the size of my dataset (145).
# The reason, as explained in Task 5 description, the use of stratified shuffle split
# cross validation.
# Also, for this initial classifier, there is a large number of false negatives,
# and accordingly, a bad Recall.

# *******************************************************************
# I will now try a Decision Tree using GridSearchCV for systematic tuning
# Ref: http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html

## CODE:
print "CLASSIFIER:"
print

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
DT_parameters = {'criterion':('gini','entropy'), 'min_samples_split':[4,8,12]}
dtc = DecisionTreeClassifier() # Already imported

#-----------------------[REVIEW]
# As recommended in the review, using StratifiedShuffleSplit, I will validate
# the algorithm performance using a cv object that best adapts to dataset characteristics,
# and searches for those parameters that maximize RECALL using the 'scoring' parameter
# in GridSearchCV:
#
cv = StratifiedShuffleSplit(labels,n_iter = 50,random_state = 42)
a_grid_search = GridSearchCV(dtc, param_grid = DT_parameters,cv = cv, scoring = 'recall')
a_grid_search.fit(features,labels)

print "Decision Tree Classifier"
print "Chosen parameters:", a_grid_search.best_estimator_

## pick a winner
best_clf = a_grid_search.best_estimator_ # This is the best classifier

clf = best_clf
print clf
print


# Output from tester.py: (Before picking this algorithm and tuning it with StratifiedShuffleSplit)
#GridSearchCV(cv=None, error_score='raise',
#       estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            random_state=None, splitter='best'),
#       fit_params={}, iid=True, loss_func=None, n_jobs=1,
#       param_grid={'min_samples_split': [4, 8], 'criterion': ('gini', 'entropy')},
#       pre_dispatch='2*n_jobs', refit=True, score_func=None, scoring=None,
#       verbose=0)
#	Accuracy: 0.82786	Precision: 0.37330	Recall: 0.30200	F1: 0.33389	F2: 0.31399
#	Total predictions: 14000	True positives:  604	False positives: 1014	False negatives: 1396	True negatives: 10986
#
# Interestingly Accuracy is lower, Precision also diminished, but Recall improved slightly.
# However, it seems, overall, a more balanced classifier.


# *******************************************************************
# I will try now AdaBoost with Decision Tree Classifier as weak learner:

## CODE:
"""
from sklearn.ensemble import AdaBoostClassifier
base_clf = DecisionTreeClassifier(min_samples_split=4) # From prior GridSearchCV
clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators = 20) # n_estimators chosen arbitrary
clf.fit(features, labels)
"""

# Output from tester.py:
#AdaBoostClassifier(algorithm='SAMME.R',
#          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
#            min_samples_split=4, min_weight_fraction_leaf=0.0,
#            random_state=None, splitter='best'),
#          learning_rate=1.0, n_estimators=20, random_state=None)
#	Accuracy: 0.82336	Precision: 0.29381	Recall: 0.16850	F1: 0.21417	F2: 0.18421
#	Total predictions: 14000	True positives:  337	False positives:  810	False negatives: 1663	True negatives: 11190
#
# In this case, this ensemble method gave worst results.

# *******************************************************************
# I will try SVC with scaling (since I will test rbf kernel).
# I will also use  selectKBest for feature selection using the feature set
# 'features_list_try_3' as 'feature_list' which has the initial 10 features
# plus the 2 new features.
# I will use Pipeline and Gridsearch:
# Ref: http://scikit-learn.org/stable/modules/pipeline.html

## CODE:
"""
features_list = ['poi','salary','total_stock_value','total_payments',
                 'restricted_stock','exercised_stock_options', 'expenses',
                 'to_messages', 'from_messages', 'from_poi_to_this_person',
                 'from_this_person_to_poi', 'fraction_from_poi', 'fraction_to_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

estimators = [('scaling', MinMaxScaler()), ('selection', SelectKBest()), ('algorithm', SVC())]
my_pipeline = Pipeline(estimators)
parameters = dict(selection__k=[4,5,6], algorithm__kernel=('linear', 'rbf'), algorithm__C=[0.1, 1, 10, 100])
clf = GridSearchCV(my_pipeline, param_grid=parameters)
clf.fit(features, labels)
"""

# Output from tester.py:
#GridSearchCV(cv=None, error_score='raise',
#       estimator=Pipeline(steps=[('scaling', MinMaxScaler(copy=True, feature_range=(0, 1))), ('selection', SelectKBest(k=10, score_func=<function f_classif at 0x000000001453AAC8>)), ('algorithm', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
#  kernel='rbf', max_iter=-1, probability=False, random_state=None,
#  shrinking=True, tol=0.001, verbose=False))]),
#       fit_params={}, iid=True, loss_func=None, n_jobs=1,
#       param_grid={'selection__k': [4, 5, 6], 'algorithm__C': [0.1, 1, 10, 100], 'algorithm__kernel': ('linear', 'rbf')},
#       pre_dispatch='2*n_jobs', refit=True, score_func=None, scoring=None,
#       verbose=0)
#	Accuracy: 0.86920	Precision: 0.57983	Recall: 0.06900	F1: 0.12332	F2: 0.08376
#	Total predictions: 15000	True positives:  138	False positives:  100	False negatives: 1862	True negatives: 12900

# Recall did very poorly since there is a large number for false negatives

# *******************************************************************
# I will use the Decision Tree Classifier

# *******************************************************************
# Evaluation for the final design: (TUNING DONE IN PREVIOUS TASK)
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
"""
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
"""

# Given the nature of the dataset I will use k-fold cross-validation
# Ref: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html
# Ref: http://www.analyticsvidhya.com/blog/2015/05/k-fold-cross-validation-simple/
from sklearn.cross_validation import KFold
from sklearn import metrics

kf = KFold(len(labels), n_folds=3)
accuracies = []
recalls = []
precisions = []
# transform lists into numpy arrays
# Ref: http://stackoverflow.com/questions/24215886/typeerror-when-attempting-cross-validation-in-sklearn
# Ref: http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.asarray.html
features_np_array = np.asarray(features)
labels_np_array = np.asarray(labels)
for train_index, test_index in kf:
    features_train, features_test = features_np_array[train_index], features_np_array[test_index]
    labels_train, labels_test = labels_np_array[train_index], labels_np_array[test_index]
    clf = clf.fit(features_train, labels_train)
    labels_predicted = clf.predict(features_test)
    acc = clf.score(features_test, labels_test)
    accuracies.append(acc)
    recall = metrics.recall_score(labels_test, labels_predicted)
    recalls.append(recall)
    precision = metrics.precision_score(labels_test, labels_predicted)
    precisions.append(precision)

# Now the results and their means:
print "Final design mean evaluations, using k=3 fold cross validation"
print "Accuracy values for each fold (not relevant since dataset is heavily unbalanced):", accuracies
#print "Mean accuracy:", np.mean(accuracies)
print "Recall values for each fold:", recalls
print "Recall:", np.mean(recalls)
print "Precisionvalues for each fold:", precisions
print "Precision:", np.mean(precisions)
print
print "Training Using second fold:"
fold_count = 1
for train_index, test_index in kf:
    features_train, features_test = features_np_array[train_index], features_np_array[test_index]
    labels_train, labels_test = labels_np_array[train_index], labels_np_array[test_index]
    if fold_count == 2:
        clf = clf.fit(features_train, labels_train)
        labels_predicted = clf.predict(features_test)
        break
    fold_count += 1

final_acc = clf.score(features_test, labels_test)
final_recall = metrics.recall_score(labels_test, labels_predicted)
final_precision = metrics.precision_score(labels_test, labels_predicted)
print "Final accuracy:", final_acc
print "Final recall:", final_recall
print "Final precision:", final_precision
print

# Since both Recall and Precision are above 0.3 I will keep this classifier

# *******************************************************************
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)



# *******************************************************************
print "*************************************************************"
print "REVIEW"
print "Importance of features for final classifier:"
for index,importance in enumerate(clf.feature_importances_):
    print "Feature:", features_list[index+1]
    print "Importance:", importance

#Importance of the new feature used:
#Feature: exercised_stock_options
#Importance: 0.0965865130105
#Feature: expenses
#Importance: 0.422405016899
#Feature: fraction_from_poi
#Importance: 0.390279461739
#Feature: restricted_stock
#Importance: 0.0907290083506

# IMPORTANCE OF NEW FEATURE:
# fraction_from_poi has the second largest value with 0.390279461739



### Feature importance with iterative process (second iteration):
#Importance of features:
#Feature: bonus
#Importance: 0.401094102404
#Feature: total_stock_value
#Importance: 0.111776484497
#Feature: long_term_incentive
#Importance: 0.0
#Feature: exercised_stock_options
#Importance: 0.131296599624
#Feature: other
#Importance: 0.355832813475
#Feature: expenses
#Importance: 0.0
### Recall and Precision:
#Final recall: 0.333333333333
#Final precision: 0.333333333333


### Feature importance with iterative process (third and final iteration):
#Importance of features:
#Feature: bonus
#Importance: 0.424737717027
#Feature: shared_receipt_with_poi
#Importance: 0.0
#Feature: exercised_stock_options
#Importance: 0.0502403018989
#Feature: other
#Importance: 0.525021981074
### Recall and Precision:
#Final recall: 0.166666666667
#Final precision: 0.25
