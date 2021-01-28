import os
import pandas as pd
import numpy as np
# Load data
def load_data(FeaturesFile):
    return pd.read_csv(FeaturesFile)
FeaturesFile = "C:\\Users\\Pengyang\\机器学习书目代码\\ML\\Data\\Feature1.csv"
OrderFeaturesRaw = load_data(FeaturesFile)
print(OrderFeaturesRaw.info())

#分割测试集和训练集
from sklearn.model_selection import StratifiedShuffleSplit #分层抽样
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

#查看属性和标签的相关性
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
print(a.head(60))
print('ok')

#标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train)
x_train_s = pd.DataFrame(scaler.transform(x_train))
x_test_s = pd.DataFrame(scaler.transform(x_test))


# 模型训练
from sklearn.ensemble import RandomForestClassifier #随机森林
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
svm_clf = SVC(gamma="scale", random_state=42, probability=True)

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],voting='soft')
voting_clf.fit(x_train_s, y_train)
voting_clf_pred = voting_clf.predict(x_test_s)
voting_clf_pred_precent = voting_clf.predict_proba(x_test_s)
#统计模型
from sklearn.metrics import accuracy_score
p1 = accuracy_score(y_test, rnd_clf_pred)
p2 = accuracy_score(y_test, bag_clf_pred)
p3 = accuracy_score(y_test, voting_clf_pred)
print(p1)
print(p2)
print(p3)


num = 0
for i in range(len(rnd_clf_pred_percent)):
    if max(rnd_clf_pred_percent[i])<0.8:
        print(i)
        num = num+ 1
        print(rnd_clf_pred_percent[i])
print(x_test.head())
print(num)
num = 0
for i in range(len(voting_clf_pred_precent)):
    if max(voting_clf_pred_precent[i])<0.8:
        print(i)
        num = num+ 1
        print(voting_clf_pred_precent[i])
print(num)