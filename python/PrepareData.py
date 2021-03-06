import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


# Load data
def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing_path = "C:\\Users\\Pengyang\\机器学习书目代码\\handson-ml2\\datasets\\housing"
housing = load_housing_data(housing_path)
print(housing.info())
# split train and test
from sklearn.model_selection import StratifiedShuffleSplit
housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# split Feature and Labels
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
print(type(housing_labels))
# drop ocean_proximity
housing_num = housing.drop("ocean_proximity", axis=1)
housing_num = housing_num.drop("income_cat", axis=1)



if False:
    # split train and test
    from sklearn.model_selection import StratifiedShuffleSplit
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # split Feature and Labels
    housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    # add new feature
    #housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    #housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    #housing["population_per_household"]=housing["population"]/housing["households"]

    print(housing.info())
    print(housing.describe())

    # Data Clean
    # drop ocean_proximity
    housing_num = housing.drop("ocean_proximity", axis=1)
    print(housing_num.info())

    # 使用中位数填充缺失值
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing_num)  # count median number of all feature
    X = imputer.transform(housing_num)  # replace null feature with median  of all feature
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing.index) # trans data
    print(housing_tr.info())

    # 处理特征中的非数字值，分类值，字符串等
    housing_cat = housing[["ocean_proximity"]]
    print(housing_cat.info())
    from sklearn.preprocessing import OneHotEncoder
    cat_encoder = OneHotEncoder()

    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    housing_cat_1hot.toarray()
    #print(housing_cat_1hot)
    print('----------------')
    print(type(housing_cat_1hot))

#转换器 pipline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        print(type(X))
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
print(20*'#')
print(housing_num.info())
print(20*'#')
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

old_num_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

old_cat_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])
old_full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", old_num_pipeline),
        ("cat_pipeline", old_cat_pipeline),
    ])
print(20*'@')

housingd = housing.drop("income_cat", axis=1)
print(housingd.info())
old_housing_prepared = old_full_pipeline.fit_transform(housingd)
#print(old_housing_prepared)
print(old_housing_prepared.shape)
print('ok')

# 训练和评估模型
lin_reg = LinearRegression()
lin_reg.fit(old_housing_prepared, housing_labels) # the model

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(old_housing_prepared, housing_labels)

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(old_housing_prepared, housing_labels)

from sklearn.svm import SVR
svm_reg = SVR(kernel="linear")
svm_reg.fit(old_housing_prepared, housing_labels)

if False:
    # 使用训练的模型进行预测
    some_data = housingd.iloc[:5]
    print(some_data.info())
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = old_full_pipeline.transform(some_data)
    pridects = lin_reg.predict(some_data_prepared)
    print(pridects)
    print(some_labels)

#评估模型
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(old_housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

print(20*'-')
housing_predictions_tree = svm_reg.predict(old_housing_prepared)
tree_lin_mse = mean_squared_error(housing_labels,housing_predictions_tree)
tree_lin_rmse = np.sqrt(tree_lin_mse)
print(tree_lin_rmse)

#交叉验证评估 tree_reg
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, old_housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

print(display_scores(tree_rmse_scores))

# 调整参数（超参数）， 网格搜索,适合少量参数组合
if False:
    from sklearn.model_selection import GridSearchCV
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
      ]
    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)
    grid_search.fit(old_housing_prepared, housing_labels)
    print(grid_search.best_params_)

# 调整参数（超参数），随机搜索,适合大量参数组合
if True:
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint
    param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    rnd_search.fit(old_housing_prepared, housing_labels)
    feature_importances = grid_search.best_estimator_.feature_importances_
    print(feature_importances)

    # 查看模型在测试集上的表现
    final_model = rnd_search.best_estimator_

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)
