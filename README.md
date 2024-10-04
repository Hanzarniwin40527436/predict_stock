# Stock Price Prediction using LSTM

This project demonstrates how to use a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical data. The model is trained to predict the next day’s closing price using the past 60 days of stock prices.

## Project Overview

The script loads historical stock price data (from Yahoo Finance), preprocesses it, and trains an LSTM model to predict future stock prices. After training, the model is used to predict stock prices for the test set, and the results are visualized with actual vs. predicted prices.

### Features:
- Download historical stock price data using the `yfinance` library.
- Normalize the data for better performance with the LSTM model.
- Create training and testing datasets using a sliding window of past 60 days.
- Build and train an LSTM model using the TensorFlow Keras API.
- Visualize actual vs. predicted prices.

## Requirements

To run this project, you need the following Python libraries:

- `numpy`
- `pandas`
- `yfinance`
- `scikit-learn`
- `tensorflow`
- `matplotlib`

You can install them using pip:

```bash
pip install numpy pandas yfinance scikit-learn tensorflow matplotlib
