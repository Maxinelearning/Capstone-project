# Stock Price Prediction with LSTM and Artificial Rabbit Optimization

## Project Overview

This project implements a Long Short-Term Memory (LSTM) model to predict stock prices based on historical data. In this project, we retrieve 5 major US tech company financial data for 10 years starting from 1/1/2025 - 31/12/2024. Then we train the model to take 90 days of historical record to predict the price in the next 7 days. Inspired by the work of Burak Gülmez (2023), we replicate his model architecture and use Artificial Rabbit Optimization (ARO) algorithm to optimize of the model hyperparameters.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Results](#results)
- [License](#license)


## Installation

To run this project, you need to have Python and the necessary libraries installed. You can install the required packages using pip. Below is a list of the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```

### Required Libraries
numpy: For numerical operations.
pandas: For data manipulation and analysis.
matplotlib: For plotting and visualization.
scikit-learn: For preprocessing and evaluation metrics.
tensorflow: For building and training the LSTM model.
keras: For high-level neural networks API (included with TensorFlow).


### Notes

- Make sure your python version and pip meet the requirement for installing tensorflow. For more details, please visit https://www.tensorflow.org/install/pip
- It is recommended to use a virtual environment for running this model


## Data Preparation
The data preparation involves the following steps:

1. retrieve the data from yfinance
2. Load historical stock price data.
3. Normalize the data for better model performance.
4. Create sequences of features and targets using the create_sequences function. This function generates input-output pairs based on specified previous and prediction timesteps.


## Model Training
The LSTM model is built using TensorFlow/Keras and consists of:

1. Input layer
2. One or more LSTM layers
3. Dense layer
4. Output layer

The model is trained on the prepared dataset for a specified number of epochs using a defined batch size.

## Hyperparameter Optimization
Hyperparameters such as the number of neurons, dropout rate, learning rate, and more are optimized using the ARO algorithm. The ARO algorithm iteratively adjusts these parameters to minimize the defined objective function, such as Mean Squared Error (MSE).

## Results
After training, the model's performance is evaluated on a validation set and displayed in a graph showing the model loss over epochs. The loss graphs suggest the model is reliable and there is no overfitting issues.


## Contributing
Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.


## License
This project is licensed under the MIT License.
