import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib
import tempfile


df = pd.read_csv("Data_for_MlFlow.csv")

# Prepare features
categorical_features = [
    "Category", "Sub-Category", "Ship Mode",
    "Market", "Region", "Segment", "Season"
]

numeric_features = [
    "Discount", "Profit",
    "Year", "Month", "Day", "Day_of_Week", "WeekOfYear",
    "Is_Weekend", "IsMonthStart", "IsMonthEnd",
    "Sales_lag_1", "Sales_lag_7", "Sales_lag_30",
    "Qty_lag_1", "Qty_lag_7", "Qty_lag_30",
    "Sales_roll_mean_7", "Sales_roll_std_7",
    "Qty_roll_mean_7", "Qty_roll_std_7",
    "Sales_diff_1", "Qty_diff_1",
]

X = df[numeric_features + categorical_features]
y_sales = df["Sales"]
y_qty = df["Quantity"]

# define models
models = {
    "final_Sales_xgb": "Models_joblib/final_sales_xgb_model.joblib",
    "final_qty_xgb": "Models_joblib/final_qty_xgb_model.joblib",
    "Sales_XGBoost": "Models_joblib/Sales_XGBoost_model.joblib",
    "Sales_RandomForest": "Models_joblib/Sales_RandomForest_model.joblib",
    "Sales_DecisionTree": "Models_joblib/Sales_DecisionTree_model.joblib",
    "Sales_KNN": "Models_joblib/Sales_KNN_model.joblib",
    "Sales_AdaBoost": "Models_joblib/Sales_AdaBoost_model.joblib",
    "Quantity_XGBoost": "Models_joblib/Quantity_XGBoost_model.joblib",
    "Quantity_RandomForest": "Models_joblib/Quantity_RandomForest_model.joblib",
    "Quantity_DecisionTree": "Models_joblib/Quantity_DecisionTree_model.joblib",
    "Quantity_KNN": "Models_joblib/Quantity_KNN_model.joblib",
    "Quantity_AdaBoost": "Models_joblib/Quantity_AdaBoost_model.joblib"
}

#MLflow logging
mlflow.set_tracking_uri("mlruns")

for model_name, model_path in models.items():
    print(f"Running experiment for {model_name}...")
    if "Sales" in model_name:
        mlflow.set_experiment("Sales-Models")
    else:
         mlflow.set_experiment("Quantity-Models")

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        continue

    with mlflow.start_run(run_name=model_name):
        loaded_model = joblib.load(model_path)
        
        if model_name in ["final_Sales_xgb", "final_qty_xgb"]:
            model = loaded_model
            # Determine target variable and get predictions
            if "Sales" in model_name:
                y_true = y_sales
                target_name = "Sales"
            else:
                y_true = y_qty
                target_name = "Quantity"
            
            y_pred = model.predict(X)

            # Calculate regression metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            # Log metrics
            mlflow.log_metric("RMSE", rmse) 
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("MSE", mse)
            # Create prediction vs actual plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.6)
            max_val = max(y_true.max(), y_pred.max())
            min_val = min(y_true.min(), y_pred.min())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            plt.xlabel(f"Actual {target_name}")
            plt.ylabel(f"Predicted {target_name}")
            plt.title(f"Actual vs Predicted - {model_name}")

            with tempfile.TemporaryDirectory() as tmpdir:
               plot_path = os.path.join(tmpdir, f"{model_name}_scatter.png")
               plt.savefig(plot_path)
               mlflow.log_artifact(plot_path)
            plt.close()

            signature = infer_signature(X, y_pred)
            mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=model_name,
            signature=signature )

        else:
            model = loaded_model["model"]
            metrics = loaded_model["metrics"]
            rmse = metrics.get("rmse")
            mae = metrics.get("mae")
            r2 = metrics.get("r2")
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)

        # Log parameters if available
        if hasattr(model, 'get_params'):
          params = model.get_params()
          for i, (key, value) in enumerate(params.items()):
            if i >= 10:  
               break
            if value is not None:
               mlflow.log_param(key, value)

print("All experiments completed successfully.")