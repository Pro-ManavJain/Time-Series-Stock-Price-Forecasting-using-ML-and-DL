**Time-Series Stock Price Forecasting using Machine Learning & Deep Learning**

Overview:

Forecasting stock prices is one of the most challenging applications of data science due to the dynamic and sequential nature of financial markets. Traditional machine learning models often struggle to capture temporal dependencies, while deep learning models are specifically designed to learn patterns from sequential data.

This project explores and compares the performance of Linear Regression, Random Forest, LSTM (Long Short-Term Memory), and GRU (Gated Recurrent Unit) models for predicting stock closing prices using historical market data.

Using Apple's (AAPL) stock data collected from Yahoo Finance, the project evaluates how well each model learns market behavior and predicts future prices based on historical trends.

Project Objectives:

1) Collect real-world stock market data.
2) Build a forecasting pipeline for time-series prediction.
3) Compare traditional Machine Learning and Deep Learning approaches.
4) Analyze the strengths and limitations of each model.
5) Evaluate prediction accuracy using standard regression metrics.

Dataset:

The dataset is fetched directly from Yahoo Finance using the yfinance library.

Stock Used:
Apple Inc. (AAPL)

Time Period:
Start Date: January 1, 2018, End Date: January 1, 2024

Features:
| Feature | Description                     |
| ------- | ------------------------------- |
| Open    | Opening price                   |
| High    | Highest price of the day        |
| Low     | Lowest price of the day         |
| Volume  | Number of shares traded         |
| Close   | Closing price (Target Variable) |

Dataset Size:
1509 Rows, 5 Features

Workflow:

1. Data Collection:
  Historical stock data is downloaded using:

  yf.download("AAPL")

  The downloaded dataset includes daily stock prices and trading volume.

2. Data Preprocessing:

  The following preprocessing steps are performed:

  Multi-Index Column Handling:

  Yahoo Finance returns a MultiIndex format.

  Example:
  ('Close', 'AAPL')
  ('High', 'AAPL')
  ('Low', 'AAPL')

  These are converted into single-level columns:
  Close
  High
  Low
  Open
  Volume

3. Feature Selection:

Input Features:
Open
High
Low
Volume

Target Variable:
Close

4. Train-Test Split:

Since stock prices are sequential data, random shuffling is avoided.

The dataset is split chronologically:
80% Training Data, 20% Testing Data

This ensures that future information does not leak into training.

5. Feature Scaling:

Financial data contains values of different magnitudes.

To normalize the data:
MinMaxScaler()

is applied separately to:
Input Features
Target Variable

Resulting range:
0 → 1

Models Implemented
1. Linear Regression
2. Random Forest Regressor
3. LSTM (Long Short-Term Memory)
4. GRU (Gated Recurrent Unit)

Visualization:

The project includes visual comparisons between::

LSTM Predictions vs Actual Prices
1. Tracks overall market trends.
2. Captures long-term movement effectively.

GRU Predictions vs Actual Prices

1. Produces predictions closer to actual values.
2. Responds more quickly to short-term fluctuations.
