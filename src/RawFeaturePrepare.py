from sklearn.base import BaseEstimator , TransformerMixin
import pandas as pd
import numpy as np 

class RawFeaturePreparer(BaseEstimator , TransformerMixin):

    def __init__(self):
        self.street_counts = {}
        self.zip_to_city = {}
        self.zip_to_state = {}
        self.city_to_zip = {}
        self.city_to_state = {}
        self.house_size_upper_limit = None
        self.acre_lot_upper_limit = None

        print("Executing Raw Feature Preparer ")

    def fit(self , X , y = None):

        # ----- Feature Engineering -----
    
        self.street_counts = X['street'].value_counts().to_dict()
    
        # ----- Fill Missing Values -----

        zip_city_state_map_df = X[['zip_code', 'city', 'state']].dropna(subset=['zip_code', 'city', 'state'])
    
        self.zip_to_city = zip_city_state_map_df.groupby('zip_code')['city'].first().to_dict()
        self.zip_to_state = zip_city_state_map_df.groupby('zip_code')['state'].first().to_dict()
    
        self.city_to_zip = zip_city_state_map_df.groupby('city')['zip_code'].first().to_dict()
        self.city_to_state = zip_city_state_map_df.groupby('city')['state'].first().to_dict()

        self.house_size_upper_limit = X['house_size'].quantile(0.98)
    
        self.acre_lot_upper_limit = X['acre_lot'].quantile(0.97)

        return self
    
    def transform(self , X):
        X_out = X.copy()

        # ----- Standardize Columns -----
    
        X_out['prev_sold_date'] = pd.to_datetime(X_out['prev_sold_date'] , yearfirst= True , errors= 'coerce')

        # ----- Feature Engineering -----
    
        X_out['is_previously_sold'] = X_out['prev_sold_date'].notna().astype(int)
        X_out['street_density'] = X_out['street'].map(self.street_counts).fillna(0)
    
        X_out = X_out.drop(columns=['street' , 'prev_sold_date'] , axis = 1 , errors = 'ignore')

        # ----- Fill Missing Values -----
    
        X_out['city'] = X_out['city'].fillna(X_out['zip_code'].map(self.zip_to_city))
        X_out['state'] = X_out['state'].fillna(X_out['zip_code'].map(self.zip_to_state))
        X_out['zip_code'] = X_out['zip_code'].fillna(X_out['city'].map(self.city_to_zip))
        X_out['state'] = X_out['state'].fillna(X_out['city'].map(self.city_to_state))

        

        x_out_shape = X_out.shape[0]

        # ----- Handle Outliers -----
    
        X_out['house_size'] = X_out['house_size'].clip(upper = self.house_size_upper_limit , lower = 400.0)
        X_out['acre_lot'] = X_out['acre_lot'].clip(upper = self.acre_lot_upper_limit , lower = 0.02)


        X_out['bed'] = X_out['bed'].clip(upper = 12 , lower = 0)
        X_out['bath'] = X_out['bath'].clip(upper = 10 , lower = 0)
    
        outliers_removed = ((x_out_shape - X_out.shape[0])/x_out_shape) * 100
        print(f"Rows Percent Dropped in removing outliers : {outliers_removed :.2f} %")


        print("Ending Raw Feature Preparer ")

        return X_out