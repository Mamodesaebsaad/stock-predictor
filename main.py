from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import pandas as pd
import requests
from datetime import datetime, timedelta
import yfinance as yf
import json
from prophet import Prophet

# Load config
with open("config.json", "r") as f:
    CONFIG = json.load(f)

app = FastAPI()

class StockRequest(BaseModel):
    ticker: str
    period: int = 1  # Number of time units
    unit: str = "days"  # "minutes", "hours", "days", "months", "years"

def fetch_data_yfinance(ticker: str) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    if df.empty:
        raise ValueError("No data found with yfinance.")
    df = df[['Close']].reset_index()
    df.rename(columns={"Close": "close", "Date": "date"}, inplace=True)
    return df

def fetch_data_custom_api(ticker: str) -> pd.DataFrame:
    url = CONFIG["custom_api_url"].replace("{ticker}", ticker)
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to fetch data from custom API.")
    
    data = response.json()
    
    if "Time Series (Daily)" not in data:
        raise ValueError("Unexpected response format from API.")
    
    timeseries = data["Time Series (Daily)"]
    
    records = []
    for date_str, daily_data in timeseries.items():
        records.append({
            "date": pd.to_datetime(date_str),
            "close": float(daily_data["4. close"])
        })

    df = pd.DataFrame(records)
    df.sort_values("date", inplace=True)
    return df


def train_model(df: pd.DataFrame) -> float:
    df['days'] = (df['date'] - df['date'].min()).dt.days
    model = LinearRegression()
    model.fit(df[['days']], df['close'])
    tomorrow = df['days'].max() + 1
    prediction = model.predict([[tomorrow]])
    predicted_price = float(prediction[0])
    
    predicted_date = df['date'].min() + pd.Timedelta(days=tomorrow)
    return {
        "predicted_price": predicted_price,
        "predicted_date": predicted_date
    }
    # return float(prediction[0])


def train_model_prophet(df: pd.DataFrame, periods: int = 1, freq: str = 'D') -> dict:
    # Create a copy to avoid modifying original DataFrame
    prophet_df = df.copy()
    
    # Ensure we have the right columns
    if 'date' not in prophet_df.columns or 'close' not in prophet_df.columns:
        raise ValueError("DataFrame must contain 'date' and 'close' columns")
    
    # Prophet needs 'ds' and 'y' columns
    prophet_df = prophet_df[['date', 'close']].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Ensure proper data types
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
    
    # Remove any NaN values
    prophet_df = prophet_df.dropna()
    
    # Sort by date and reset index
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    # Ensure we have enough data
    if len(prophet_df) < 2:
        raise ValueError("Need at least 2 data points for Prophet")
    
    # Configure seasonality based on prediction timeframe
    yearly_seasonality = freq in ['M', 'Y']  # Enable for months/years
    weekly_seasonality = freq in ['D', 'H']  # Enable for days/hours
    daily_seasonality = freq == 'H'  # Enable for hours
    
    # Initialize and fit Prophet model
    model = Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality
    )
    model.fit(prophet_df)

    # Forecast specified periods into the future
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # Get all future predictions (not just the last one)
    future_predictions = forecast.iloc[-periods:]
    
    # Return multiple predictions for longer periods or single prediction for short periods
    if periods == 1:
        prediction_row = forecast.iloc[-1]
        return {
            "predicted_price": float(prediction_row["yhat"]),
            "lower_bound": float(prediction_row["yhat_lower"]),
            "upper_bound": float(prediction_row["yhat_upper"]),
            "predicted_date": prediction_row["ds"]
        }
    else:
        predictions = []
        for _, row in future_predictions.iterrows():
            predictions.append({
                "predicted_price": float(row["yhat"]),
                "lower_bound": float(row["yhat_lower"]),
                "upper_bound": float(row["yhat_upper"]),
                "predicted_date": row["ds"]
            })
        return {"predictions": predictions}

def convert_unit_to_freq(unit: str) -> str:
    """Convert time unit to pandas frequency string"""
    unit_mapping = {
        "minutes": "T",  # T for minutes in pandas
        "hours": "H",
        "days": "D", 
        "months": "M",
        "years": "Y"
    }
    return unit_mapping.get(unit.lower(), "D")

def validate_prediction_request(period: int, unit: str) -> None:
    """Validate prediction request parameters"""
    valid_units = ["minutes", "hours", "days", "months", "years"]
    if unit.lower() not in valid_units:
        raise ValueError(f"Invalid unit. Must be one of: {valid_units}")
    
    if period <= 0:
        raise ValueError("Period must be greater than 0")
    
    # Set reasonable limits
    limits = {
        "minutes": 1440,  # Max 24 hours in minutes
        "hours": 168,     # Max 7 days in hours  
        "days": 365,      # Max 1 year in days
        "months": 60,     # Max 5 years in months
        "years": 10       # Max 10 years
    }
    
    max_period = limits.get(unit.lower(), 365)
    if period > max_period:
        raise ValueError(f"Period too large. Maximum {max_period} {unit} allowed")

@app.get("/")
def root():
    return {"status": "ok", "message": "Stock Predictor API is running."}

@app.post("/predict")
def predict(request: StockRequest):
    try:
        ticker = request.ticker.upper()
        period = request.period
        unit = request.unit.lower()
        
        # Validate request
        validate_prediction_request(period, unit)
        
        # Get data
        if CONFIG["data_source"] == "yfinance":
            df = fetch_data_yfinance(ticker)
        else:
            df = fetch_data_custom_api(ticker)

        # Convert unit to pandas frequency
        freq = convert_unit_to_freq(unit)
        
        # Get predictions
        prediction_data = train_model_prophet(df, periods=period, freq=freq)
        
        # Format response based on single vs multiple predictions
        if "predictions" in prediction_data:
            # For multiple predictions, show summary + individual predictions
            predictions = []
            total_predictions = len(prediction_data["predictions"])
            
            # Calculate average confidence interval
            avg_lower = sum(pred["lower_bound"] for pred in prediction_data["predictions"]) / total_predictions
            avg_upper = sum(pred["upper_bound"] for pred in prediction_data["predictions"]) / total_predictions
            avg_price = sum(pred["predicted_price"] for pred in prediction_data["predictions"]) / total_predictions
            
            # Get first and last prediction for summary
            first_pred = prediction_data["predictions"][0]
            last_pred = prediction_data["predictions"][-1]
            
            # Calculate actual start time based on current time for minutes/hours
            current_time = datetime.now()
            if unit == "minutes":
                actual_start = current_time + timedelta(minutes=1)
                actual_end = current_time + timedelta(minutes=period)
            elif unit == "hours":
                actual_start = current_time + timedelta(hours=1)
                actual_end = current_time + timedelta(hours=period)
            else:
                # For days, months, years use the prediction dates
                actual_start = first_pred["predicted_date"]
                actual_end = last_pred["predicted_date"]
            
            # Simplified response with summary
            return {
                "ticker": ticker,
                "prediction_period": f"{period} {unit}",
                "summary": {
                    "average_predicted_price": round(avg_price, 2),
                    "average_confidence_interval": {
                        "lower": round(avg_lower, 2),
                        "upper": round(avg_upper, 2)
                    },
                    "price_range": {
                        "lowest": round(min(pred["lower_bound"] for pred in prediction_data["predictions"]), 2),
                        "highest": round(max(pred["upper_bound"] for pred in prediction_data["predictions"]), 2)
                    },
                    "total_predictions": total_predictions
                },
                "time_range": {
                    "start_date": actual_start.strftime("%Y-%m-%d %H:%M:%S") if unit in ["minutes", "hours"] else actual_start.strftime("%Y-%m-%d"),
                    "end_date": actual_end.strftime("%Y-%m-%d %H:%M:%S") if unit in ["minutes", "hours"] else actual_end.strftime("%Y-%m-%d")
                },
                "explanation": {
                    "note": f"Prophet model trained on daily data predicting {unit}-level changes",
                    "accuracy": "Lower accuracy for short time intervals (minutes/hours) due to limited granular data",
                    "recommendation": "Use daily or longer periods for more reliable predictions"
                },
                "prediction_timestamp": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
                "data_range_start": df['date'].min().strftime("%Y-%m-%d"),
                "data_range_end": df['date'].max().strftime("%Y-%m-%d"),
                "detailed_predictions": predictions[:10] if len(predictions) > 10 else predictions  # Show only first 10 for brevity
            }
        else:
            # Single prediction - keep existing format
            date_format = "%Y-%m-%d %H:%M:%S" if unit in ["minutes", "hours"] else "%Y-%m-%d"
            
            return {
                "ticker": ticker,
                "prediction_period": f"{period} {unit}",
                "predicted_close_price": round(prediction_data["predicted_price"], 2),
                "confidence_interval": {
                    "lower": round(prediction_data["lower_bound"], 2),
                    "upper": round(prediction_data["upper_bound"], 2)
                },
                "predicted_date": prediction_data["predicted_date"].strftime(date_format),
                "explanation": {
                    "model": "Facebook Prophet with seasonality detection",
                    "confidence": "68% of actual values should fall within the confidence interval",
                    "data_basis": f"Prediction based on last 60 days of daily closing prices"
                },
                "prediction_timestamp": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
                "data_range_start": df['date'].min().strftime("%Y-%m-%d"),
                "data_range_end": df['date'].max().strftime("%Y-%m-%d")
            }
            
    except Exception as e:
        return {"error": str(e)}

# Additional date manipulation example
def get_previous_day(date_str: str) -> datetime:
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return date_obj - timedelta(days=1)

# Example usage:
# previous_day = get_previous_day("2025-08-22")
# print(previous_day)

