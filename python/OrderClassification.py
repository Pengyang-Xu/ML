#-*- coding = utf-8
import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
# Load data
def load_data(FeaturesFile):
    return pd.read_csv(FeaturesFile)
try:
    FeaturesFile = "D:\\MyGit\\ML\Data\\Feature3.csv"
    OrderFeaturesRaw = load_data(FeaturesFile)
except:
    FeaturesFile = "D:\\MyGit\\ML\Data\\Feature3.csv"
    OrderFeaturesRaw = load_data(FeaturesFile)
print(OrderFeaturesRaw.info())


from sklearn.model_selection import StratifiedShuffleSplit 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(OrderFeaturesRaw, OrderFeaturesRaw["Classification"]):
    train_set = OrderFeaturesRaw.loc[train_index]
    test_set = OrderFeaturesRaw.loc[test_index]

print(train_set.info())
print(OrderFeaturesRaw['Classification'].value_counts()/len(OrderFeaturesRaw))
print(train_set['Classification'].value_counts()/len(train_set))

x_train = train_set.drop("Classification", axis=1)
y_train = train_set['Classification'].copy()

x_test = test_set.drop("Classification", axis=1)
y_test = test_set['Classification'].copy()
print(x_train)


from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

OrderFeaturesRawClass = OrderFeaturesRaw[['Classification']]
print(OrderFeaturesRawClass.head(10))
encoder = OrdinalEncoder()
encoder2 = OneHotEncoder()
OrderFeaturesRawClass_encoder = encoder.fit_transform(OrderFeaturesRawClass)

print(type(OrderFeaturesRawClass_encoder))
print(OrderFeaturesRaw.head(10))

OrderFeaturesRaw_f = OrderFeaturesRaw.drop('Classification', axis=1)
#print(OrderFeaturesRaw_f.info())
OrderFeaturesRaw_f['classfy']=OrderFeaturesRawClass_encoder

corr_matrix = OrderFeaturesRaw_f.corr()
a=corr_matrix["classfy"].sort_values(ascending=False)

print(a)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train)
x_train_s = pd.DataFrame(scaler.transform(x_train))
x_test_s = pd.DataFrame(scaler.transform(x_test))


from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,n_jobs=1)
rnd_clf.fit(x_train_s,y_train)
rnd_clf_pred = rnd_clf.predict(x_test_s)
rnd_clf_pred_percent = rnd_clf.predict_proba(x_test_s)
#bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=500,max_samples=100, bootstrap=True, random_state=42)
bag_clf.fit(x_train_s, y_train)
bag_clf_pred = bag_clf.predict(x_test_s)


# voting
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf.fit(x_train_s, y_train)

# save model
joblib.dump(rnd_clf, 'D:\\MyGit\\ML\Models\\rnd_clf')
# load model
clf3 = joblib.load('D:\\MyGit\\ML\Models\\rnd_clf')

print(rnd_clf.feature_importances_)
importtance = rnd_clf.feature_importances_
print(type(importtance.tolist()))
c = importtance.tolist()
c.sort()
for i in c:
    print(i)

svm_clf = SVC(gamma="scale", random_state=42, probability=True)

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='soft')
voting_clf.fit(x_train_s, y_train)
voting_clf_pred = voting_clf.predict(x_test_s)
voting_clf_pred_precent = voting_clf.predict_proba(x_test_s)

joblib.dump(rnd_clf, 'D:\\MyGit\\ML\Models\\voting_clf')

voting_clf_pred_train = voting_clf.predict(x_train_s)
voting_clf_pred_precent_train = voting_clf.predict_proba(x_train_s)
#
from sklearn.metrics import accuracy_score
p1 = accuracy_score(y_test, rnd_clf_pred) #randomforst
p2 = accuracy_score(y_test, bag_clf_pred) # BaggingClassifier
p3 = accuracy_score(y_test, voting_clf_pred) # voting classifier (LogisticRegression, RandomForestClassifier, svm)
print(p1)
print(p2)
print(p3)

# 
from sklearn.metrics import precision_score, recall_score

print(precision_score(y_test, voting_clf_pred,average='weighted'))
print(recall_score(y_test, voting_clf_pred,average='weighted'))

print(clf3.predict(x_test_s))
if False:
    print(30*'#')
    pred_index = []
    for i in range(len(voting_clf_pred_precent)):
        if max(voting_clf_pred_precent[i])<0.8:
            pred_index.append([i, voting_clf_pred_precent[i][0] , voting_clf_pred_precent[i][1],voting_clf_pred_precent[i][2]])
            
    print(pred_index)
    
    x_test_index=[]
    
    nfd2 = open("D:\\MyGit\\ML\Data\\x_test_seqindex.csv",'w')
    for i in x_test['L'].index.tolist():
        x_test_index.append(i)
    print(voting_clf_pred)
    
    for i in pred_index:
        print(i)
    
        nfd2.write(str(x_test_index[i[0]])+'\t'+str(i[1])+'\t'+str(i[2])+'\t'+str(i[3])+'\n')
        
    #############################################################
    print(30*'#')
    pred_index_train = []
    for i in range(len(voting_clf_pred_precent_train)):
        if max(voting_clf_pred_precent_train[i])<0.8:
            pred_index_train.append([i, voting_clf_pred_precent_train[i][0] , voting_clf_pred_precent_train[i][1],voting_clf_pred_precent_train[i][2]])
            
    print(pred_index_train)
    
    x_train_index=[]
    nfd3 = open("D:\\MyGit\\ML\Data\\x_train_seqindex.csv",'w')
    for i in x_train['L'].index.tolist():
        x_train_index.append(i)
    print(voting_clf_pred_train)
    
    for i in pred_index_train:
        print(i)
    
        nfd3.write(str(x_train_index[i[0]])+'\t'+str(i[1])+'\t'+str(i[2])+'\t'+str(i[3])+'\n')
