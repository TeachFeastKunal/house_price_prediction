import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator , TransformerMixin
from utils import log_transform

class HierarchialGroupImputer(BaseEstimator , TransformerMixin):
    def __init__(self):
        self.global_med_ = {}
        self.state_med_ = {}
        self.state_city_med_ = {}

        print("Executing Data Imputer (HierarchialGroupImputer) ")


    def fit(self , X , y = None):

        self.global_med_['house_size'] = X['house_size'].median()
        self.global_med_['acre_lot'] = X['acre_lot'].median()
        self.global_med_['bed'] = X['bed'].median()
        self.global_med_['bath'] = X['bath'].median()

        # State Level Medians

        self.state_med_['house_size'] = X.groupby('state')['house_size'].median()
        self.state_med_['acre_lot'] = X.groupby('state')['acre_lot'].median()

        # State-City Level Medians

        self.state_city_med_['house_size'] = X.groupby(['state' , 'city'])['house_size'].median()
        self.state_city_med_['acre_lot'] = X.groupby(['state' , 'city'])['acre_lot'].median()

        return self

    def transform(self , X):
        x_out = X.copy()

        x_out['bed'] = x_out['bed'].fillna(self.global_med_['bed'])
        x_out['bath'] = x_out['bath'].fillna(self.global_med_['bath'])

        state_city_index = x_out.set_index(['state' , 'city']).index

        # State-City Level Imputation
        imputed_house_size = state_city_index.map(self.state_city_med_['house_size'])

        # State Level Imputation
        imputed_house_size = pd.Series(imputed_house_size , index = x_out.index).fillna(x_out['state'].map(self.state_med_['house_size']))

        # Global 

        imputed_house_size = imputed_house_size.fillna(self.global_med_['house_size'])

        x_out['house_size'] = x_out['house_size'].fillna(imputed_house_size)

        # ACRE-LOT

        # State-City Level Imputation
        imputed_acre_lot = state_city_index.map(self.state_city_med_['acre_lot'])

        # State Level Imputation
        imputed_acre_lot = pd.Series(imputed_acre_lot , index = x_out.index).fillna(x_out['state'].map(self.state_med_['acre_lot']))

        # Global 

        imputed_acre_lot = imputed_acre_lot.fillna(self.global_med_['acre_lot'])

        x_out['acre_lot'] = x_out['acre_lot'].fillna(imputed_acre_lot)

        log_transform_cols = ['house_size' , 'acre_lot']
        x_out = log_transform(x_out , cols = log_transform_cols)

        cols_to_drop = ['house_size' , 'acre_lot']
        x_out = x_out.drop(columns=cols_to_drop , axis = 1 , errors = 'ignore')

        print("Ending Data Imputer (HierarchialGroupImputer) ")

        return x_out


class CategoricalEncoder(BaseEstimator , TransformerMixin):
    def __init__(self):
        self.state_target_mean_ = None
        self.city_target_mean_ = None
        self.global_target_mean_ = None
        self.train_cols_ = None

        print("Executing Data Encoder (CategoricalEncoder) ")
        
    def fit(self , X , y = None):
    
        self.state_target_mean_ = y.groupby(X['state']).mean().to_dict()
        self.city_target_mean_ = y.groupby(X['city']).mean().to_dict()
        self.global_target_mean_ = y.mean()

        self.state_target_mean_.pop(np.nan, None)
        self.city_target_mean_.pop(np.nan, None)

        x_dummy = pd.get_dummies(X , columns = ['status'] , drop_first = True)
        dummy_cols = [c for c in x_dummy.columns if c.startswith('status_')]

        for col in dummy_cols:
            x_dummy[col] = x_dummy[col].astype(int)

        x_dummy['state_encoded'] = X['state'].map(self.state_target_mean_).fillna(self.global_target_mean_)
        x_dummy['city_encoded'] = X['city'].map(self.city_target_mean_).fillna(self.global_target_mean_)

        cols_to_drop = ['state' , 'city' , 'prev_sold_date' , 'street' , 'prev_sold_date' , 'brokered_by' , 'zip_code']
        x_dummy = x_dummy.drop(columns=cols_to_drop , axis = 1 , errors = 'ignore')

        self.train_cols_ = x_dummy.columns
        return self


    def transform(self , X):
        x_out = pd.get_dummies(X, columns=['status'], drop_first=True)
        dummy_cols = [c for c in x_out.columns if c.startswith('status_')]
        for col in dummy_cols:
            x_out[col] = x_out[col].astype(int)

        x_out['state_encoded'] = x_out['state'].map(self.state_target_mean_).fillna(self.global_target_mean_)
        x_out['city_encoded'] = x_out['city'].map(self.city_target_mean_).fillna(self.global_target_mean_)

        x_final = x_out.reindex(columns = self.train_cols_ , fill_value = 0)

        print("Ending Data Encoder (CategoricalEncoder) ")

        return x_final