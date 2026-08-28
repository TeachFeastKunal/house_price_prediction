import joblib
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import load_data , null_pct , transform_target
from impute_catencode import HierarchialGroupImputer , CategoricalEncoder
from RawFeaturePrepare import RawFeaturePreparer


MODEL_PATH = Path(r'HousePricePrediction\models\pipeline.joblib')
DATA_PATH = Path(r'HousePricePrediction\data\realtor-data.zip.csv')

def main():
    df = load_data(DATA_PATH)

    null_pct_df = null_pct(df)
    print(f"Null Count Of Columns is : \n {null_pct_df}")

    df = transform_target(df , target_col = 'price')

    drop_cols = ['price' , 'log_price' ]

    train_df , test_df = train_test_split(df , test_size = 0.2 , random_state = 42)

    x_train = train_df.drop(drop_cols , axis = 1 , errors = 'ignore')
    y_train = train_df['log_price']

    x_test = test_df.drop(drop_cols , axis = 1 , errors = 'ignore')
    y_test = test_df['log_price']

    param_grid = {'n_estimators': 750,
                'learning_rate': 0.03,
                'max_depth': 18,
                'subsample': 0.8,
                'colsample_bytree': 1.0,
                'reg_lambda': 0.5,
                'max_bin' : 1024
                }

    pipeline = Pipeline([
        ('Feature_prep' , RawFeaturePreparer()) ,
        ('imputer' , HierarchialGroupImputer()),
        ('catencoder' , CategoricalEncoder()),
        ('scaler' , StandardScaler()),
        ('xgb' , XGBRegressor(
            tree_method = 'hist' , 
            n_jobs = -1,
            random_state = 42,
            **param_grid
        )) ,
    ],  verbose = True )

    pipeline.fit(x_train , y_train)
    Path(MODEL_PATH).parent.mkdir(parents = True , exist_ok = True)

    joblib.dump({
        "model" : pipeline, 
        "x_test" : x_test , 
        "y_test" : y_test 
    }, MODEL_PATH)

    print(f"Model Saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
