# Experimental Deep Learning for Stock Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Experimental-orange)

## 📌 Project Overview
This repository contains a comparative study of various Deep Learning architectures (LSTM, ANN, Hybrid Models) applied to financial time-series forecasting. The primary goal is to predict stock price movements (specifically AAPL) by analyzing historical **Log Returns**.

Unlike standard implementations, this project emphasizes **Data Leakage Prevention** and robust validation using **Time Series Cross-Validation**.

## 🚀 Key Features
* **Hybrid Architecture:** Combines LSTM (for temporal feature extraction) with Dense/ANN layers (for regression).
* **Leak-Free Methodology:** Strict separation of Training, Validation, and Test sets based on temporal order to prevent look-ahead bias.
* **Robust Preprocessing:** Uses Log-Returns ($ln(P_t / P_{t-1})$) instead of raw prices to ensure data stationarity.
* **Advanced Validation:** Implements `TimeSeriesSplit` (Rolling Window Cross-Validation) rather than random shuffling.
* **Real-Scale Metrics:** All evaluation metrics (MSE, RMSE, MAPE) are calculated by back-transforming predictions to the **original price scale**.

## 🧠 Models Implemented
The repository includes experiments with the following architectures:
* **LSTM + ANN (Hybrid):** 2-Layer LSTM followed by a Multi-Layer Perceptron.
* **Pure LSTM:** Standard Long Short-Term Memory network.
* **Bi-Directional LSTM:** (Optional - if you added this)
* **GRU:** Gated Recurrent Units for comparison.

## 📊 Performance Benchmarks
Comparison of different models on the `AAPL` dataset (Final Test Set):

| Model Architecture | Epochs | RMSE (Price) | MAPE (%) | R² Score | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Simple ANN** | 50 | 2.45 | 1.85% | 0.65 | Baseline |
| **Standard LSTM** | 100 | 1.89 | 1.12% | 0.82 | Good |
| **Hybrid LSTM+ANN** | **50** | **1.34** | **0.89%** | **0.91** | **Best Performance** |

*(Note: The values above are examples. Please update them with your actual test results.)*

## ⚙️ Technical Methodology

### 1. Data Preprocessing
Raw closing prices are converted to Log Returns to stabilize the mean and variance:
$$r_t = \ln(\frac{P_t}{P_{t-1}})$$

### 2. Windowing
A sliding window approach (default `WINDOW_SIZE=30`) is used to create sequence data for the LSTM.

### 3. Model Training (TimeSeriesSplit)
The model is trained using 5-fold cross-validation designed for time series:
```text
Fold 1: [Train: Jan-Mar] -> [Test: Apr]
Fold 2: [Train: Jan-Apr] -> [Test: May]
...
