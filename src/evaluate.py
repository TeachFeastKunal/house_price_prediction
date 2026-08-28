import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error , root_mean_squared_error , r2_score

MODEL_PATH = Path(r'HousePricePrediction\models\pipeline.joblib')

def main():

    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    x_test = model_data['x_test']
    y_test = model_data['y_test']
    predict_price = model.predict(x_test)

    dollars_actual = np.expm1(y_test)
    dollars_pred = np.expm1(predict_price)

    log_mae = mean_absolute_error(y_test, predict_price)
    log_rmse = root_mean_squared_error(y_test, predict_price)
    log_r2 = r2_score(y_test, predict_price)

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

if __name__ == "__main__":
    main()
