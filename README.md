# Deep Learning Stock Prediction Research & Optimization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20%26%20Experimental-orange?style=for-the-badge)

## 📌 Project Overview
This repository serves as a **Deep Learning laboratory** for financial time-series forecasting. The primary goal is not just to implement a single algorithm, but to **explore, compare, and optimize a wide range of neural network architectures** to predict stock price movements (specifically AAPL).

Instead of relying on a specific model, this project conducts a **comparative analysis** across different methodologies—ranging from simple Recurrent Networks to complex Hybrid structures and Attention-based mechanisms—to identify the most robust solution for volatile market data.

## 🚀 Key Features & Engineering Approach
* **Multi-Model Architecture:** Experiments with various configurations including Pure LSTM, GRU, Hybrid Models (LSTM+ANN), and Custom Attention Layers.
* **Leak-Free Methodology:** Strict separation of Training, Validation, and Test sets based on temporal order to prevent look-ahead bias.
* **Robust Preprocessing:** Uses **Log-Returns** ($ln(P_t / P_{t-1})$) instead of raw prices to ensure data stationarity and better convergence.
* **Advanced Validation:** Implements `TimeSeriesSplit` (Rolling Window Cross-Validation) rather than random shuffling.
* **Real-Scale Metrics:** All evaluation metrics (MSE, RMSE, MAPE) are calculated by back-transforming predictions to the **original price scale**.

## 🧠 Models & Architectures Explored
This project investigates the performance of diverse deep learning strategies:

* **Hybrid Models:** Combining sequence processing (LSTM/GRU) with Dense (ANN) layers for enhanced feature regression.
* **Attention Mechanisms:** Custom layers designed to weigh the importance of specific time-steps in the input sequence.
* **Recurrent Architectures:** Standard implementations of LSTM, Bi-Directional LSTM, and GRU for benchmarking.
* **Optimization Experiments:** Testing various hyperparameters (Layer depth, Hidden units, Dropout rates) to minimize overfitting.

## 📊 Performance Benchmarks
### Top Performing Architectures
*Results obtained from the final leak-free test set (AAPL Dataset).*

| Model Architecture | Epochs | RMSE (Price) | MAPE (%) | R² Score | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Hybrid LSTM + ANN** | 50 | 2.8685 | 1.3690% | 0.9975 | High Performance |
| **Hybrid LSTM + ATTENTION** | 25 | **2.8612** | **1.3663%** | **0.9975** | **Best Accuracy** |
| **Simple LSTM** | 25 | 2.8614 | 1.3663% | 0.9975 | Baseline Benchmark |

*(Note: The 'Hybrid LSTM + Attention' model achieved the lowest error rate (RMSE) on unseen test data, demonstrating better generalization capabilities.)*

## ⚙️ Technical Methodology

### 1. Data Preprocessing
Raw closing prices are converted to Log Returns to stabilize the mean and variance, which is crucial for neural network stability:
$$r_t = \ln(\frac{P_t}{P_{t-1}})$$

### 2. Windowing Strategy
A sliding window approach (default `WINDOW_SIZE=30`) is utilized to convert time-series data into supervised learning sequences.

### 3. Training & Validation (TimeSeriesSplit)
To simulate real-world trading scenarios, the models are trained using a forward-chaining cross-validation scheme:
```text
Fold 1: [Train: Jan-Mar] -> [Test: Apr]
Fold 2: [Train: Jan-Apr] -> [Test: May]
Fold 3: [Train: Jan-May] -> [Test: Jun]
...
