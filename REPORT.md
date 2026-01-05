# REPORT: Lipschitz Neural Network for Volatility Forecasting

## Dataset

S&P500 Dataset containing Open, High, Low, Close, Adj Close, Volume, Dividends and Next day changes data.

Using this dataset, we calculated the **Garman-Klass** volatility estimator defined as,

$\sigma_{GK}^2 = \frac{1}{2} \ln(\frac{H}{L})^2 - (2 \ln(2) - 1) \ln(\frac{C}{O})^2$

And we scaled the values in $[-1, 1]$, using MinMaxScaler.

## Model architecture

### Feedforward Neural Network

For the classical Feedforward Neural Network, we used the following architecture, 

- input Layer: 21 features

- first Layer: features to 64 neurons with ReLU activation function. 

- second Layer: 64 to 32 neurons with ReLU activation function. 

- third Layer: 32 to 1 neuron with Loss function Mean Squared Error (MSE).

### Lipschitz Neural Network

For the Lipschitz Neural Network, we used the following architecture, 

- input Layer: 21 features

- first Layer: 21 to 64 neurons, we apply the Spectral normalization on the linear part of the layer, and then we used GroupSort activation function

- second Layer: 64 to 32 neurons, we apply the Spectral normalization on the linear part of the layer, and then we used GroupSort activation function

- third Layer: 32 to 1 neuron, we apply the Spectral normalization on the linear part of the layer, and then we used the MSE Loss function to calculate the loss. 

## Workflow

### Data

- Calculate the Garman-Klass volatility estimator

- Create sequences to split the volatility estimator values into input and target data. The number of inputs is defined by the variable "seq_length" in the code, and the next value is the target value. 

- Scale the data using MinMaxScaler and split the dataset into train and test sets. 

- Transform the dataset into PyTorch Dataset such that the data have the right format to train our models.

### Train function

Define a function that will be used to train our models. This function also plots the training and validation set loss values, and plots the predictions of the trained model. 

### Models

- Implement the feedforward neural network as defined above, and train it using the train function.

- Then, implement the Lipschitz neural network as defined above, and train it using the same train function.

### Adversarial Attack

Finally, implement the Fast Gradient Sign Method adversarial attack, for different epsilon values, to compare the robustness and the stability of the feedforward neural network, against the Lipschitz neural network. 

## Results

### 4.2 Robustness Comparison Summary

The table below summarizes the performance of the Feed-Forward Neural Network (FNN) versus the Lipschitz Neural Network (LNN) under increasing levels of adversarial noise ($\epsilon$).

| Epsilon ($\epsilon$) | FNN Loss | LNN Loss | FNN MAE | LNN MAE |
| :--- | :--- | :--- | :--- | :--- |
| **0.000** | 0.010573 | **0.009629** | 0.002846 | **0.002586** |
| **0.010** | 0.012626 | **0.011271** | 0.003285 | **0.003052** |
| **0.050** | 0.023713 | **0.020392** | 0.004989 | **0.004801** |
| **0.100** | 0.041328 | **0.034149** | 0.007114 | **0.006761** |
| **0.150** | 0.067111 | **0.050877** | 0.009340 | **0.008536** |
| **0.200** | 0.100961 | **0.070289** | 0.011619 | **0.010175** |
| **0.300** | 0.194056 | **0.115605** | 0.016343 | **0.013296** |

**Key Insight:** As the noise intensity ($\epsilon$) increases, the error (Loss and MAE) for the standard FNN explodes much faster than for the LNN. At $\epsilon=0.300$, the FNN Loss (0.194) is nearly double the LNN Loss (0.115), demonstrating the superior stability of the Lipschitz-constrained model.

We note that the performance degradation at maximum epsilon is equal to 474.13% for the FNN, and 414.12% for the LNN. The relative improvement is then equal to 60.01%.

<img src="output.png" alt="Robustness Comparison Graph" width="1080">

## Potential improvements

As I used time series data, it would be more interesting to implement a Recurrent Neural Network, and compare it to a Lipschitz Recurrent Neural Network. 