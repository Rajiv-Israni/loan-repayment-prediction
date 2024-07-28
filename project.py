import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score

x = pd.read_pickle('/dataset/dataset.pkl')
target_train = np.zeros(10000)
target_test = np.zeros(5000)
xdata_train = np.zeros([10000,5])
xdata_test = np.zeros([5000,5])

c_id = x[0]

for i in range(15000):
    c_id[i] = c_id[i][1]

for i in np.arange(0, 10000):
    transactions = x[2][i][1]
    target_train[i] = x[6][i][1]
    loan = x[4][i][1]

    count_pos_trainset = 0
    count_neg_trainset = 0
    
    for k in range(len(transactions)):
        if transactions[k]>0:
            count_pos_trainset+=1
        else:
            count_neg_trainset+=1
    threshold_trainset = 0.50 * loan
    
    xdata_train[i, 0] = np.sum(transactions)
    xdata_train[i, 1] = threshold_trainset
    xdata_train[i, 2] = count_pos_trainset
    xdata_train[i, 3] = count_neg_trainset
    xdata_train[i, 4] = loan

xdata_train[np.isnan(xdata_train)] = 0

k = 10000

for j in np.arange(0, 5000):
    transactions1 = x[2][j+k][1]
    loan1 = x[4][j+k][1]
    
    count_pos_testset = 0
    count_neg_testset = 0
    for l in range(len(transactions1)):
        if transactions1[l]>0:
            count_pos_testset+=1
        else:
            count_neg_testset+=1
    threshold_testset = 0.50 * loan1
    
    xdata_test[j, 0] = np.sum(transactions1)
    xdata_test[j, 1] = threshold_testset
    xdata_test[j, 2] = count_pos_testset
    xdata_test[j, 3] = count_neg_testset
    xdata_test[j, 4] = loan1

xdata_test[np.isnan(xdata_test)] = 0

params = {
        "n_estimators" : [1, 10, 25, 50, 75, 100, 200], 
        "criterion" : ["gini", "entropy"], 
        "max_depth" : [1, 2, 3, 4, 5], 
        "min_samples_split" : [2, 4, 6], 
        "min_samples_leaf" : [1, 2, 3, 4], 
        "max_leaf_nodes" : [4, 5, 8, 10]
        }

classifier = RandomForestClassifier()
random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)
random_search.fit(xdata_train, target_train)
print(random_search.best_estimator_)
print(random_search.best_params_)

# so we get best estimator as RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                      # max_depth=5, max_features='auto', max_leaf_nodes=10,
                      # min_impurity_decrease=0.0, min_impurity_split=None,
                      # min_samples_leaf=2, min_samples_split=6,
                      # min_weight_fraction_leaf=0.0, n_estimators=50,
                      # n_jobs=None, oob_score=False, random_state=None,
                      # verbose=0, warm_start=False)

final_classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=5, max_features='auto', max_leaf_nodes=10,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=6,
                       min_weight_fraction_leaf=0.0, n_estimators=50,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
final_classifier.fit(xdata_train, target_train)
probabilities = final_classifier.predict_proba(xdata_test)

probs_isdef = probabilities[:,1]
probs_isdef = probs_isdef.reshape(5000,1)

very_dangerous = 0
safe = 0

for g in range(len(probs_isdef)):
    
    if probs_isdef[g]>0.90:
        very_dangerous+=1
    elif probs_isdef[g]<0.05:
        safe+=1
print(r'very dangerous to lend to is:',very_dangerous)
print(r'safe to lend to is:',safe)

#therefore no subset of very dangerous and safe

index = []
m = 10000
for a in np.arange(0, 5000):
    index.append(x[0][a+m][1])

ids = np.asarray(index)
ids = ids.reshape(5000,1)


res = np.concatenate((ids, probs_isdef), axis=1)

scores_randomforest = cross_val_score(final_classifier, xdata_train, target_train, cv=10,error_score='raise-deprecating')
print(r'average score of model = ', np.mean(scores_randomforest))

import csv

def write_csv(file_path, y_list):
    solution_rows = [('id', 'category')] + [(i, y) for (i, y) in enumerate(y_list)]
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(solution_rows)

def output_submission_csv(output_file_path, y_test):
    write_csv(output_file_path, y_test)


sample = c_id
submission = pd.DataFrame({"ID": sample[10000:15000],"IsDefault": probabilities[:,1]})
submission.to_csv('output.txt',index=False, header=None)