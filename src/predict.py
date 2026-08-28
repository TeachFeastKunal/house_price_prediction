import joblib
from pathlib import Path
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_absolute_error , root_mean_squared_error , r2_score
from utils import transform_target
MODEL_PATH = Path(r'HousePricePrediction\models\pipeline.joblib')

def eval_predict_price(df : pd.DataFrame , y_test: pd.Series):

    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']

    pred = model.predict(df)
    dollars_actual = np.expm1(y_test)
    dollars_pred = np.expm1(pred)

    log_mae = mean_absolute_error(y_test, pred)
    log_rmse = root_mean_squared_error(y_test, pred)
    log_r2 = r2_score(y_test, pred)

    mae = mean_absolute_error(dollars_actual, dollars_pred)
    rmse = root_mean_squared_error(dollars_actual, dollars_pred)
    r2 = r2_score(dollars_actual, dollars_pred)

    print("================ GRADIENT BOOSTING REGRESSOR METRICS ================")
    print("\nIn Log\n")
    print(f"Mean Absolute Error  : {log_mae:.4f}")
    print(f"Root Mean Square Error : {log_rmse:.4f}")
    print(f"R2 Score    : {log_r2:.4f}")
    print("\nIn Dollars\n")
    print(f"Mean Absolute Error  : {mae:.4f}")
    print(f"Root Mean Square Error : {rmse:.4f}")
    print(f"R2 Score    : {r2:.4f}")

    
    
sample_df = pd.DataFrame({
    'brokered_by': [103378.0, 52140.0, np.nan, 84210.0, 31450.0],
    'status': ['for_sale', 'sold', 'for_sale', 'sold', 'for_sale'],
    'price': [365000.0, 620000.0, 195000.0, 1150000.0, 340000.0],
    'bed': [3.0, 4.0, 2.0, 5.0, np.nan],                
    'bath': [2.0, 2.5, 1.0, 4.0, 2.0],
    'acre_lot': [0.18, 0.45, np.nan, 1.85, 0.25],        
    'street': [1962661.0, 841235.0, 1962661.0, 948210.0, np.nan],
    'city': ['Austin', 'Seattle', 'Detroit', 'San Francisco', np.nan], 
    'state': ['Texas', 'Washington', 'Michigan', 'California', 'Texas'],
    'zip_code': [78701.0, 98101.0, 48201.0, 94102.0, 78704.0],
    'house_size': [1650.0, 2400.0, 950.0, np.nan, 1850.0], 
    'prev_sold_date': ['2018-05-12', '2021-11-03', np.nan, '2015-08-20', np.nan]
})

sample_df = transform_target(sample_df , 'price')
X_sample = sample_df.drop(columns=['price' , 'log_price'])
y_test_sample = sample_df['log_price'] 

if __name__ == "__main__":
    eval_predict_price(X_sample , y_test_sample)