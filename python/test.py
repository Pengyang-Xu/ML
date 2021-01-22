import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:,rooms_ix ] / X[: household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# this is a test
# test2

def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing_path = "C:\\Users\\penxu\\HandON_ML\\handson-ml-master\\datasets\\housing"
housing = load_housing_data(housing_path)

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

# show the datasets
if False:
    print(housing.head(6))
    
    print(housing.info())
    #print(housing.describe())
    #housing.hist(bins=50, figsize=(20,15))
    #plt.show()

# data clean
if False:
    #median = housing["total_bedrooms"].median()
    #housing["total_bedrooms"].fillna(median)
    
    imputer = Imputer(strategy="median") # get inputer object
    housing_num = housing.drop ("ocean_proximity", axis=1) # drop the ocean_proximity row, not the int or float

    imputer.fit(housing_num) # fit inputer to housing_num datasets
    #imputer.statistics_
    #housing_num.median().values
    X = imputer.transform(housing_num) # transform median to null
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index = list(housing.index.values)) # trans numpy array to pandas DataFrame
    housing_cat = housing["ocean_proximity"]
    encoder = LabelBinarizer()
    housing_cat_1hot = encoder.fit_transform(housing_cat)
    print(housing_cat_1hot)

if True:
    housing_num = housing.drop ("ocean_proximity", axis=1) # drop the ocean_proximity row, not the int or float
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder()),
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
    housing_prepared = full_pipeline.fit_transform(housing)
    print(housing_prepared.head())