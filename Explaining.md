# Stock Price Prediction: Flexible Time Period Forecasting

This document explains the stock predictor API that uses Facebook Prophet for forecasting stock prices across multiple time horizons with confidence intervals.

---

## 🎯 New Features: Flexible Time Periods

Our API now supports predictions for various time periods:
- **Minutes**: 1-1440 minutes (up to 24 hours)
- **Hours**: 1-168 hours (up to 7 days)
- **Days**: 1-365 days (up to 1 year)
- **Months**: 1-60 months (up to 5 years)
- **Years**: 1-10 years

### API Request Format
```json
{
    "ticker": "AAPL",
    "period": 5,
    "unit": "hours"
}
```

---

## Approach 1: Linear Regression (Legacy Baseline)

### Step-by-Step Breakdown

#### 1. Collect Historical Data

You collect the past 60 days of closing prices for a specific stock (e.g., AAPL). Each record includes:
- The date
- The closing price of the stock on that date

Think of this as a table of points on a graph.

#### 2. Convert Dates to Numbers

Machine learning models can't "understand" date strings, so you convert dates into numbers representing how many days ago each record is.

Example:
- June 23 → 0
- June 24 → 1
- ...
- August 22 → 60

Now your dataset is a sequence of points:
```
(day_number, closing_price)
```

#### 3. Fit a Line (Linear Regression)

You use linear regression to find the best-fitting straight line through those points. This means finding two numbers:
- **Slope (m):** how much the price changes per day
- **Intercept (b):** the starting point of the line

Mathematically, the line looks like:
```
price = m * day + b
```

The algorithm calculates `m` and `b` such that the sum of squared errors between the line and the actual prices is minimized (Ordinary Least Squares, OLS).

#### 4. Predict the Next Day

Once the line is fitted, you use it to predict the price for tomorrow:
```
predicted_price = m * 61 + b
```

This gives a point estimate: your model's best guess for the stock price tomorrow based on the past trend.

**Limitation**: Only predicts next day, no flexibility for different time periods.

---

## Approach 2: Facebook Prophet (Current Implementation)

### Enhanced Prophet with Flexible Time Periods

Prophet is now configured to handle multiple time horizons with appropriate seasonality settings:

#### Seasonality Configuration by Time Unit

| Time Unit | Daily Seasonality | Weekly Seasonality | Yearly Seasonality |
|-----------|------------------|-------------------|-------------------|
| **Minutes** | ❌ | ❌ | ❌ |
| **Hours** | ✅ | ✅ | ❌ |
| **Days** | ❌ | ✅ | ❌ |
| **Months** | ❌ | ❌ | ✅ |
| **Years** | ❌ | ❌ | ✅ |

#### Frequency Mapping
```python
unit_mapping = {
    "minutes": "T",  # T for minutes in pandas
    "hours": "H",    # H for hours
    "days": "D",     # D for days
    "months": "M",   # M for months
    "years": "Y"     # Y for years
}
```

#### Validation Limits
To prevent unreasonable requests:
- **Minutes**: Maximum 1,440 (24 hours)
- **Hours**: Maximum 168 (7 days)
- **Days**: Maximum 365 (1 year)
- **Months**: Maximum 60 (5 years)
- **Years**: Maximum 10 years

---

## 📊 API Response Formats

### Single Period Prediction
For `period = 1`:
```json
{
    "ticker": "AAPL",
    "prediction_period": "1 days",
    "predicted_close_price": 175.23,
    "confidence_interval": {
        "lower": 170.45,
        "upper": 180.01
    },
    "predicted_date": "2025-08-23",
    "prediction_timestamp": "22-Aug-2025 18:14:00"
}
```

### Multiple Period Predictions
For `period > 1`:
```json
{
    "ticker": "AAPL",
    "prediction_period": "5 hours",
    "predictions": [
        {
            "predicted_close_price": 175.23,
            "confidence_interval": {
                "lower": 170.45,
                "upper": 180.01
            },
            "predicted_date": "2025-08-22 19:00:00"
        },
        {
            "predicted_close_price": 176.45,
            "confidence_interval": {
                "lower": 171.20,
                "upper": 181.70
            },
            "predicted_date": "2025-08-22 20:00:00"
        }
        // ... more predictions
    ],
    "prediction_timestamp": "22-Aug-2025 18:14:00"
}
```

---

## 🕐 Time Period Examples

### Short-term Predictions (Minutes/Hours)
**Use Case**: Intraday trading, quick decisions
```json
{
    "ticker": "AAPL",
    "period": 30,
    "unit": "minutes"
}
```
**Output**: 30 predictions, one for each minute
**Date Format**: `"2025-08-22 18:30:00"`

### Medium-term Predictions (Days)
**Use Case**: Swing trading, weekly planning
```json
{
    "ticker": "AAPL", 
    "period": 7,
    "unit": "days"
}
```
**Output**: 7 predictions, one for each day
**Date Format**: `"2025-08-29"`

### Long-term Predictions (Months/Years)
**Use Case**: Investment planning, portfolio allocation
```json
{
    "ticker": "AAPL",
    "period": 12,
    "unit": "months"
}
```
**Output**: 12 predictions, one for each month
**Date Format**: `"2026-08-22"`

---

## 🔧 Technical Implementation

### Data Preparation Process
1. **Fetch 60 days** of historical closing prices
2. **Clean and validate** data (remove NaN values)
3. **Convert to Prophet format** (`ds` and `y` columns)
4. **Configure seasonality** based on prediction timeframe
5. **Fit Prophet model** with appropriate parameters

### Prediction Generation
```python
# Create future dataframe for specified periods
future = model.make_future_dataframe(periods=period, freq=freq)

# Generate forecasts
forecast = model.predict(future)

# Extract future predictions only
future_predictions = forecast.iloc[-periods:]
```

### Error Handling and Validation
```python
def validate_prediction_request(period: int, unit: str) -> None:
    # Check valid units
    # Check period > 0
    # Check within reasonable limits
    # Raise ValueError if invalid
```
---

## 🔍 Understanding Your Results

### Why Are Minute Predictions Similar?

When you requested 30 minutes of AAPL predictions, you got very similar prices ($232.18) for each minute. Here's why:

#### The Data Limitation Issue
- **Training Data**: Prophet was trained on 60 days of **daily** closing prices
- **Prediction Request**: You asked for **minute-level** predictions
- **Result**: Limited granular information leads to flat predictions

#### What the Numbers Mean

Your result shows:
```json
{
    "predicted_close_price": 232.18,  // Same for all minutes
    "confidence_interval": {
        "lower": 225.34,              // Uncertainty range
        "upper": 239.08
    }
}
```

**Interpretation**:
- **$232.18**: Prophet's best guess based on daily trends
- **$225.34 - $239.08**: 68% confidence range (about ±$7)
- **Similar values**: No minute-level patterns in daily training data

### When Predictions Work Best

| Time Period | Data Granularity | Prediction Quality |
|-------------|------------------|-------------------|
| **Minutes** | Daily → Minutes | ❌ Poor (flat predictions) |
| **Hours** | Daily → Hours | ⚠️ Limited (some variation) |
| **Days** | Daily → Days | ✅ Good (natural fit) |
| **Months** | Daily → Months | ✅ Good (trend + seasonality) |

### Improved Response Format

The new API response includes:
- **Summary**: Average predictions and ranges
- **Explanation**: Why results look this way
- **Recommendation**: Better time periods to use
- **Limited Details**: Only first 10 predictions shown

### Example Interpretation

For your 30-minute AAPL prediction:
- **What it shows**: Stock likely to stay around $232
- **Confidence**: Could range from $225-$239 
- **Reality**: Minute-to-minute prices will vary much more
- **Recommendation**: Use daily/weekly predictions instead

### Better Use Cases by Time Period

#### ✅ Good: Daily Predictions
```json
{
    "ticker": "AAPL",
    "period": 7,
    "unit": "days"
}
```
**Result**: Meaningful daily price forecasts with real trend information

#### ✅ Good: Monthly Predictions  
```json
{
    "ticker": "AAPL",
    "period": 6,
    "unit": "months"
}
```
**Result**: Strategic trend analysis with seasonal patterns

#### ⚠️ Limited: Minute Predictions
```json
{
    "ticker": "AAPL",
    "period": 30,
    "unit": "minutes"
}
```
**Result**: Flat predictions due to data granularity mismatch

### Key Takeaways

1. **Match time scales**: Use daily data for daily+ predictions
2. **Understand limitations**: Minutes/hours have high uncertainty
3. **Focus on trends**: Prophet excels at longer-term patterns
4. **Use confidence intervals**: They show prediction reliability

**Remember**: The flatter the predictions over short intervals, the less reliable they are for that timeframe.

---

## 🧠 How Stock Prediction Actually Works (Simple Version)

### The Basic Idea
Imagine you're trying to guess tomorrow's weather by looking at the past 60 days. That's essentially what our stock predictor does, but with stock prices instead of weather.

### Step-by-Step Process

1. **Look at History**: We collect 60 days of Apple stock prices (like $230, $232, $229, $235...)

2. **Find Patterns**: The computer looks for trends like:
   - "Stock usually goes up on Mondays"
   - "There's a gradual upward trend over time"
   - "Prices tend to cycle every few weeks"

3. **Make a Smart Guess**: Based on these patterns, it predicts: *"Tomorrow's price will likely be $235, but could range from $230 to $240"*

4. **Add Uncertainty**: Since the future is uncertain, it gives you a range instead of one exact number

### Real Example
If you ask for AAPL's price in 7 days:
- **Past data**: Last 60 days show prices between $225-$240
- **Trend**: Generally going up by about $0.50 per day  
- **Prediction**: "$238 in 7 days, but could be anywhere from $235-$241"

### Why It's Not Perfect
- **Missing information**: Doesn't know about news, earnings, or market crashes
- **Pattern assumption**: Assumes past patterns will continue (often not true)
- **Time sensitivity**: Better at predicting weeks/months than minutes/hours

### The Bottom Line
Our predictor is like a very smart calculator that spots patterns in price history and projects them forward—but it can't predict surprises, just trends.

---

## 📐 The Math Behind It (Simple Calculations)

### What the Computer Actually Does

Let's say you have Apple stock prices for the last 5 days:
```
Day 1: $230
Day 2: $232  
Day 3: $229
Day 4: $235
Day 5: $238
```

### Step 1: Find the Average Trend
```
Price change per day = (Final price - Starting price) ÷ Number of days
= ($238 - $230) ÷ 4 days = $2 per day
```

### Step 2: Calculate Tomorrow's Base Prediction
```
Tomorrow's price = Today's price + Average trend
= $238 + $2 = $240
```

### Step 3: Add Uncertainty (Confidence Interval)
The computer looks at how much prices actually varied from the trend:
```
Price variations: +$2, -$3, +$6, +$3 (compared to perfect trend line)
Average variation = About ±$3.50

So prediction becomes:
- Best guess: $240
- Range: $236.50 to $243.50
```

### Real Prophet Formula (Simplified)
Prophet uses a more complex version:
```
Predicted Price = Base Trend + Weekly Pattern + Seasonal Effect + Uncertainty

Example:
$240 = $238 (trend) + $1 (Monday boost) + $0.50 (seasonal) + $0.50 (growth)
Range: $240 ± $4 (uncertainty) = $236 to $244
```

### Why Longer Predictions Are Less Accurate
```
1 day prediction:   $240 ± $4    (uncertainty grows slowly)
7 day prediction:   $254 ± $12   (uncertainty grows faster)  
30 day prediction:  $298 ± $28   (much more uncertainty)
```

The further you predict, the wider the "maybe" range becomes!

### The Key Numbers You See
- **Predicted Price**: The computer's best mathematical guess
- **Lower Bound**: Predicted price - uncertainty amount  
- **Upper Bound**: Predicted price + uncertainty amount
- **Confidence**: "68% chance the real price falls in this range"

That's it! The magic is just smart math applied to price patterns.
