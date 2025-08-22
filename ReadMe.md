# 📈 Stock Predictor API

A FastAPI-based stock price prediction service that uses advanced time series forecasting to predict next-day stock prices with confidence intervals.

## 🚀 Features

- **Dual Data Sources**: Support for both Yahoo Finance (yfinance) and custom APIs
- **Advanced Forecasting**: Facebook Prophet model with seasonality detection
- **Confidence Intervals**: Risk assessment with upper/lower bounds
- **RESTful API**: Clean, documented endpoints
- **Flexible Configuration**: JSON-based configuration system
- **Real-time Predictions**: Get tomorrow's stock price predictions instantly

## 🛠️ Tech Stack

- **Framework**: FastAPI (Python)
- **ML Models**: Facebook Prophet, Scikit-learn
- **Data Sources**: Yahoo Finance API, Custom APIs
- **Data Processing**: Pandas, NumPy
- **Time Series**: Prophet for advanced forecasting

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install fastapi uvicorn pandas scikit-learn yfinance prophet requests pydantic
   ```

3. **Create configuration file**
   ```json
   {
     "data_source": "yfinance",
     "custom_api_url": "https://api.example.com/stock/{ticker}"
   }
   ```
   Save as `config.json` in the project root.

4. **Run the server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## 🎯 API Usage

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Health Check
```http
GET /
```

**Response:**
```json
{
    "status": "ok",
    "message": "Stock Predictor API is running."
}
```

#### Stock Price Prediction
```http
POST /predict
```

**Request Body:**
```json
{
    "ticker": "AAPL"
}
```

**Response:**
```json
{
    "ticker": "AAPL",
    "predicted_close_price": 175.23,
    "confidence_interval": {
        "lower": 170.45,
        "upper": 180.01
    },
    "predicted_date": "2025-08-23",
    "prediction_timestamp": "22-Aug-2025 18:14:00",
    "data_range_start": "2025-06-23",
    "data_range_end": "2025-08-22"
}
```

### Example Usage

#### Using curl
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"ticker": "AAPL"}'
```

#### Using Python requests
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"ticker": "AAPL"}
)
print(response.json())
```

## 🔧 Configuration

### config.json
```json
{
    "data_source": "yfinance",
    "custom_api_url": "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey=YOUR_KEY"
}
```

**Options:**
- `data_source`: `"yfinance"` or `"custom_api"`
- `custom_api_url`: URL template with `{ticker}` placeholder

## 🤖 Models

### Facebook Prophet (Default)
- **Advantages**: Seasonality detection, confidence intervals, robust to outliers
- **Use Case**: Production predictions with uncertainty quantification
- **Output**: Point prediction + confidence bounds

### Linear Regression (Baseline)
- **Advantages**: Simple, fast, interpretable
- **Use Case**: Quick baseline comparisons
- **Output**: Point prediction only

## 📊 Data Sources

### Yahoo Finance (yfinance)
- **Free**: No API key required
- **Coverage**: Global stock markets
- **Limitations**: Rate limits, delayed data

### Custom APIs
- **Flexible**: Support any API format
- **Examples**: Alpha Vantage, Quandl, IEX Cloud
- **Requirements**: Must return 'date' and 'close' fields

## 🎛️ Supported Stock Tickers

Any valid stock ticker symbol:
- **US Stocks**: AAPL, GOOGL, MSFT, TSLA
- **International**: ASML.AS, 7203.T, SAP.DE
- **ETFs**: SPY, QQQ, VTI
- **Crypto**: BTC-USD, ETH-USD

## 📈 How It Works

1. **Data Collection**: Fetches last 60 days of closing prices
2. **Preprocessing**: Cleans and formats data for Prophet
3. **Model Training**: Fits Prophet model with trend + seasonality
4. **Prediction**: Forecasts next trading day with confidence intervals
5. **Response**: Returns formatted prediction with metadata

## ⚠️ Limitations

- **Past Performance ≠ Future Results**: All predictions based on historical patterns
- **Market Events**: Cannot predict news, earnings, or black swan events
- **60-Day Window**: Limited historical context
- **Daily Predictions**: Next day only, not long-term forecasting

## 🔍 Error Handling

Common errors and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| "No data found" | Invalid ticker | Verify ticker symbol |
| "Failed to fetch" | API/Network issue | Check connection/API limits |
| "Need at least 2 data points" | Insufficient data | Try different ticker |

## 🚦 Development

### Project Structure
```
stock-predictor/
├── main.py              # FastAPI application
├── config.json          # Configuration file
├── Explaining.md         # Technical documentation
├── README.md            # This file
└── requirements.txt     # Dependencies (optional)
```

### Adding New Models
1. Create model function in `main.py`
2. Return dict with required keys: `predicted_price`, `predicted_date`
3. Update `/predict` endpoint to use new model

### Testing
```bash
# Start server
uvicorn main:app --reload

# Test health endpoint
curl http://localhost:8000/

# Test prediction
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"ticker": "AAPL"}'
```

## 📝 License

This project is for educational and research purposes. Not intended for actual trading decisions.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📞 Support

For questions or issues:
- Check the `Explaining.md` file for technical details
- Review API documentation above
- Test with known working tickers (AAPL, MSFT)

---

**⚠️ Disclaimer**: This tool is for educational purposes only. Do not use for actual trading decisions. Stock markets are unpredictable and past performance does not guarantee future results.