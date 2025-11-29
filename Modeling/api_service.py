from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sales & Quantity Prediction API")

# Load models
try:
    sales_model = joblib.load("Models_joblib/final_sales_xgb_model.joblib")
    quantity_model = joblib.load("Models_joblib/final_qty_xgb_model.joblib")
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    sales_model = None
    quantity_model = None

# Load historical data
try:
    historical_data = pd.read_csv("Data_for_MLFlow.csv")
    historical_data["Order Date"] = pd.to_datetime(historical_data["Order Date"])
    historical_data = historical_data.sort_values("Order Date").reset_index(drop=True)
    logger.info("Historical data loaded successfully")
except Exception as e:
    logger.error(f"Error loading historical data: {e}")
    historical_data = None

class PredictionRequest(BaseModel):
    order_date: str
    category: str
    sub_category: str
    ship_mode: str
    market: str
    region: str
    segment: str
    season: str
    discount: float

class PredictionResponse(BaseModel):
    predicted_sales: float
    predicted_quantity: float
    features_used: dict

# Cache for computed features to avoid redundant calculations
feature_cache = {}
def calculate_features_for_date(target_date, df):
    """Calculate all features for a given date, filling missing days if needed"""
    target_date = pd.to_datetime(target_date)
    # Return cached features if available
    if target_date in feature_cache:
        return feature_cache[target_date].copy()
    
    current_date = df['Order Date'].max()
    df = df.copy()

    # Fill data for missing dates
    while current_date < target_date:
        next_date = current_date + timedelta(days=1)

        # Mask for lags
        def lag_mean(column, days):
            mask = (df['Order Date'] >= current_date - timedelta(days=days-1)) & (df['Order Date'] <= current_date)
            return df.loc[mask, column].mean() if not df.loc[mask, column].empty else 0.0

        def rolling_stats(column, days):
            mask = (df['Order Date'] >= current_date - timedelta(days=days-1)) & (df['Order Date'] <= current_date)
            data = df.loc[mask, column]
            return data.mean() if not data.empty else 0.0, data.std() if len(data) > 1 else 0.0

        sales_lag_1 = df.loc[df['Order Date'] == current_date, 'Sales'].values[0] if not df.loc[df['Order Date'] == current_date, 'Sales'].empty else 0.0
        sales_lag_7 = lag_mean('Sales', 7)
        sales_lag_30 = lag_mean('Sales', 30)

        qty_lag_1 = df.loc[df['Order Date'] == current_date, 'Quantity'].values[0] if not df.loc[df['Order Date'] == current_date, 'Quantity'].empty else 0.0
        qty_lag_7 = lag_mean('Quantity', 7)
        qty_lag_30 = lag_mean('Quantity', 30)

        sales_roll_mean_7, sales_roll_std_7 = rolling_stats('Sales', 7)
        qty_roll_mean_7, qty_roll_std_7 = rolling_stats('Quantity', 7)

        # Differences
        prev_sales = df.loc[df['Order Date'] == current_date - timedelta(days=1), 'Sales']
        prev_qty = df.loc[df['Order Date'] == current_date - timedelta(days=1), 'Quantity']
        sales_diff_1 = sales_lag_1 - float(prev_sales.values[0]) if not prev_sales.empty else 0.0
        qty_diff_1 = qty_lag_1 - float(prev_qty.values[0]) if not prev_qty.empty else 0.0

        new_row = {
            'Order Date': next_date,
            'Sales_lag_1': sales_lag_1,
            'Sales_lag_7': sales_lag_7,
            'Sales_lag_30': sales_lag_30,
            'Qty_lag_1': qty_lag_1,
            'Qty_lag_7': qty_lag_7,
            'Qty_lag_30': qty_lag_30,
            'Sales_roll_mean_7': sales_roll_mean_7,
            'Sales_roll_std_7': sales_roll_std_7,
            'Qty_roll_mean_7': qty_roll_mean_7,
            'Qty_roll_std_7': qty_roll_std_7,
            'Sales_diff_1': sales_diff_1,
            'Qty_diff_1': qty_diff_1
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        # store in cache
        feature_cache[next_date] = new_row
        current_date = next_date

    # Features for the target date
    features = df[df['Order Date'] == target_date]
    if features.empty:
        raise ValueError(f"No features could be generated for date {target_date}")
    
    
    features = features.iloc[0].to_dict()

    # Add datetime features
    features.update({
        'Year': target_date.year,
        'Month': target_date.month,
        'Day': target_date.day,
        'Day_of_Week': target_date.weekday(),
        'WeekOfYear': target_date.isocalendar().week,
        'Is_Weekend': 1 if target_date.weekday() >= 5 else 0,
        'IsMonthStart': 1 if target_date.is_month_start else 0,
        'IsMonthEnd': 1 if target_date.is_month_end else 0
    })
    # Automatically calculate Profit using historical average margin
    average_margin = (historical_data['Profit'] / historical_data['Sales']).mean()
    features['Profit'] = features['Sales_lag_1'] * average_margin

    return features

@app.get("/")
def read_root():
    return {"message": "Sales & Quantity Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        if historical_data is None or sales_model is None or quantity_model is None:
            raise HTTPException(status_code=500, detail="Models or historical data not loaded")

        # Generate features
        calculated_features = calculate_features_for_date(request.order_date, historical_data)

        input_data = {
            'Discount': [request.discount],
            'Profit': [calculated_features.get('Profit', 0.0)],
            'Category': [request.category],
            'Sub-Category': [request.sub_category],
            'Ship Mode': [request.ship_mode],
            'Market': [request.market],
            'Region': [request.region],
            'Segment': [request.segment],
            'Season': [request.season],
            'Year': [calculated_features.get('Year', 2024)],
            'Month': [calculated_features.get('Month', 1)],
            'Day': [calculated_features.get('Day', 1)],
            'Day_of_Week': [calculated_features.get('Day_of_Week', 0)],
            'WeekOfYear': [calculated_features.get('WeekOfYear', 1)],
            'Is_Weekend': [calculated_features.get('Is_Weekend', 0)],
            'IsMonthStart': [calculated_features.get('IsMonthStart', 0)],
            'IsMonthEnd': [calculated_features.get('IsMonthEnd', 0)],
            'Sales_lag_1': [calculated_features.get('Sales_lag_1', 0.0)],
            'Sales_lag_7': [calculated_features.get('Sales_lag_7', 0.0)],
            'Sales_lag_30': [calculated_features.get('Sales_lag_30', 0.0)],
            'Qty_lag_1': [calculated_features.get('Qty_lag_1', 0.0)],
            'Qty_lag_7': [calculated_features.get('Qty_lag_7', 0.0)],
            'Qty_lag_30': [calculated_features.get('Qty_lag_30', 0.0)],
            'Sales_roll_mean_7': [calculated_features.get('Sales_roll_mean_7', 0.0)],
            'Sales_roll_std_7': [calculated_features.get('Sales_roll_std_7', 0.0)],
            'Qty_roll_mean_7': [calculated_features.get('Qty_roll_mean_7', 0.0)],
            'Qty_roll_std_7': [calculated_features.get('Qty_roll_std_7', 0.0)],
            'Sales_diff_1': [calculated_features.get('Sales_diff_1', 0.0)],
            'Qty_diff_1': [calculated_features.get('Qty_diff_1', 0.0)]
        }
        
        input_df = pd.DataFrame(input_data)

        numeric_columns = ['Discount', 'Profit', 'Year', 'Month', 'Day', 'Day_of_Week', 'WeekOfYear', 
                          'Is_Weekend', 'IsMonthStart', 'IsMonthEnd', 'Sales_lag_1', 'Sales_lag_7', 
                          'Sales_lag_30', 'Qty_lag_1', 'Qty_lag_7', 'Qty_lag_30', 'Sales_roll_mean_7', 
                          'Sales_roll_std_7', 'Qty_roll_mean_7', 'Qty_roll_std_7', 'Sales_diff_1', 'Qty_diff_1']
        for col in numeric_columns:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(float)

        logger.info(f"Input DataFrame columns: {input_df.columns.tolist()}")
        logger.info(f"Input DataFrame dtypes: {input_df.dtypes.to_dict()}")

        # Predict
        sales_prediction = sales_model.predict(input_df)[0]
        quantity_prediction = quantity_model.predict(input_df)[0]
        
        # Ensure non-negative predictions
        sales_prediction = max(0, sales_prediction)
        quantity_prediction = max(0, quantity_prediction)

        return PredictionResponse(
            predicted_sales=float(sales_prediction),
            predicted_quantity=int(quantity_prediction),
            features_used=calculated_features
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": sales_model is not None and quantity_model is not None,
        "historical_data_loaded": historical_data is not None
    }
