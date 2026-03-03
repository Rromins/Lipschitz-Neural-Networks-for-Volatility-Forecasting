# Lipschitz Neural Network for Volatility Forecasting

> Comparing the adversarial robustness of a standard Feedforward Neural Network vs. a Lipschitz-constrained Neural Network on SPY (S&P 500) volatility prediction.

---

## Overview

Even though Deep learning models are powerful, they are prone to overfitting and instability. This project investigates whether enforcing **Lipschitz constraints** on a neural network improves its robustness to adversarial perturbations in a financial time-series context. 

Two models are trained on daily SPY (S&P 500 ETF) data to predict **realized volatility**, then subjected to **FGSM (Fast Gradient Sign Method)** attacks of increasing strength. The results show that the Lipschitz-constrained network degrades significantly less under attack.

---

## Project Structure

```
├── activations.py            # GroupSort activation function
├── normalization.py          # Spectral Norm Linear layer (power iteration)
├── models.py                 # FeedforwardNN and LipschitzNN architectures
├── adversarial_attack.py     # FGSM attack + robustness evaluation & plotting
├── lipschitz_neural_network.ipynb  # Training pipeline for the Lipschitz NN
├── SPY_daily_data.csv        # Historical SPY OHLCV data
└── output.png                # Robustness comparison plots
```

---

## Architecture

### Feedforward Neural Network (`FeedforwardNN`)

A standard 3-layer MLP with ReLU activations:

```
Input -> Linear(in, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 1)
```

### Lipschitz Neural Network (`LipschitzNN`)

A Lipschitz-constrained MLP combining two key components:

```
Input -> [SpectralNormLinear -> GroupSort] × N -> SpectralNormLinear -> Output
```

| Component | Role |
|---|---|
| **SpectralNormLinear** | Constrains each layer's Lipschitz constant via spectral normalization (power iteration) |
| **GroupSort** | Gradient-norm-preserving activation — preserves the Lipschitz bound through activations |

The **global Lipschitz constant** of the network is bounded by the product of each layer's individual constant (set via `lipschitz_const`).

---

## Key Concepts

### Spectral Normalization
Each weight matrix $W$ is rescaled by its largest singular value $\sigma(W)$:

$$\tilde{W} = \lambda \cdot \frac{W}{\sigma(W)}$$

where $\lambda$ is the target Lipschitz constant. The spectral norm $\sigma(W)$ is approximated efficiently using **Power Iteration**, updating left/right singular vector estimates $u, v$ at each forward pass.

### GroupSort Activation
Unlike ReLU (which is 1-Lipschitz but not gradient-norm-preserving), GroupSort **sorts** elements within fixed-size groups in descending order. This preserves the gradient norm, making it the ideal complement to spectral normalization for tight Lipschitz bounds.

### FGSM Attack
The Fast Gradient Sign Method perturbs inputs in the direction that maximizes the loss:

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(\theta, x, y))$$

A higher $\epsilon$ means a stronger attack.

---

## Results

![Robustness Comparison](output.png)

| Metric | Feedforward NN | Lipschitz NN |
|---|---|---|
| Loss at $\epsilon=0.0$ (clean) | 0.010 | 0.013 |
| Loss at $\epsilon=0.30$ | 0.181 | 0.064 |
| Performance degradation at $\epsilon=0.30$ | **493%** | **212%** |

The Lipschitz NN shows **~2.3× better robustness** under strong adversarial attack, with significantly less prediction scatter (bottom-right plot).

---

## Getting Started

### Prerequisites

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

### Usage

1. **Train the Lipschitz NN** — run `lipschitz_neural_network.ipynb`
2. **Compare robustness** — the adversarial evaluation is handled by `adversarial_attack.py`:

```python
from adversarial_attack import compare_model_robustness

fnn_results, lnn_results = compare_model_robustness(
    fnn_model=fnn_model,
    lnn_model=lnn_model,
    valid_loader=valid_loader,
    criterion=criterion,
    epsilon_values=[0.01, 0.05, 0.10, 0.15, 0.20, 0.30],
    scaler=scaler,
    device=device
)
```

### Instantiating the Models

```python
from models import FeedforwardNN, LipschitzNN

# Standard MLP
fnn = FeedforwardNN(in_features=10)

# Lipschitz-constrained MLP
lnn = LipschitzNN(
    input_dim=21,
    hidden_dim=[64, 32],
    output_dim=1,
    lipschitz_const=0.5,  # per-layer Lipschitz bound
    nb_iterations=10,     # power iteration steps
    group_size=2          # GroupSort group size
)
```

---

## Data

`SPY_daily_data.csv` contains historical daily OHLCV data for the SPY ETF. The target variable is **realized volatility**, computed using the Garman-Klass Volatility estimator. This is a range-based estimator that is far more efficient than close-to-close variance estimator. It has the following formula,

$$\sigma^2_{GK} = 0.5 * \ln(\frac{H}{L})^2 - (2* \ln 2 - 1) * \ln(\frac{C}{O})^2 $$

where $OHLC$ stand for *Open*, *High*, *Low*, *Close*.

Data is normalized with a `MinMaxScaler` before training; the scaler is required by the robustness evaluation to report metrics in the original price scale.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## References

- **Miyato et al.** (2018): "Spectral Normalization for Generative Adversarial Networks".
- **Boissin** (2020): "Building Lipschitz constrained networks with DEEL-LIP"
- **Béthune et al.** (2022): "Pay attention to your loss: understanding misconceptions about 1-Lipschitz neural networks"
- **Goodfellow** (2015): "Deep learning"
- **Ducotterd** (2024): "Improving Lipschitz-Constrained Neural Networks by Learning Activation Functions"
- **Cosgrove** (2018): "Spectral Normalization Explained"