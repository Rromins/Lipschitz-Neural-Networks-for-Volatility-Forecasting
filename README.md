# Lipschitz Neural Network for Volatility forecasting

## Overview

This project explores the application of a Lipschitz Neural Network to volatility forecasting. Even though Deep learning models are powerful, they are prone to overfitting and instability, especially with financial data that are noisy and chaotic. 

Enhancing Deep learning models with a $1$-Lipschitz constraint on the network layers, by using spectral normalization or Björck Orthonormalization, may improve the neural network in $3$ particular points, 

- **Robustness**: the model is less sensitive to small perturbations or outliers in market data.

- **Stability**: this constraint guarantees a bounded output change, for a bounded input change. 

- **Generalization**: the network captures the underlying market dynamics rather than memorizing noise. 

## Objectives

The objectives of this project are,

- Implement a Lipschitz Neural Network to forecast the volatility of the S&P500. 

- Compare the loss of this Lipschitz Neural Network to a classical Feedforward Neural Network, under adversarial attack, to show that the robustness and the stability are better for LNN.

- Implement the Fast Gradient Sign Method adversarial attack to evaluate the robustness of the Lipschitz Neural Network, compared to a classic Feedforward Neural Network.

## Methods

- **Garman-Klass Volatility**: this is a volatility estimator that is based on OHLC data.

- **Spectral normalization**: to ensure the Lipschitz property for the network, we implement spectral normalization using power iteration.

- **GroupSort**: rather than using the classical ReLU activation function on the Lipschitz neural network layers, we use a GroupSort activation function that is $1$-Lipschitz, and Gradient Norm Preserving (GNP).

- **Fast Gradient Sign Method attack**: this is an adversarial attack used to evaluate robustness of neural networks. It calculates the gradient of the loss function, with respect to the input data, and then adds a small perturbation in the direction of the gradient's sign.

## Key findings

In this project, we will effectively see that the Lipschitz Neural Network is more robust and stable than a Feedforward Neural Network, for regression tasks. 

## References

- **Miyato et al.** (2018): "Spectral Normalization for Generative Adversarial Networks".

- **Boissin** (2020): "Building Lipschitz constrained networks with DEEL-LIP"

- **Béthune et al.** (2022): "Pay attention to your loss: understanding misconceptions about 1-Lipschitz neural networks"

- **Goodfellow** (2015): "Deep learning"

- **Ducotterd** (2024): "Improving Lipschitz-Constrained Neural Networks by Learning Activation Functions"

- **Cosgrove** (2018): "Spectral Normalization Explained"