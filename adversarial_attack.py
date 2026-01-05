"""
Adversarial attack
"""

from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def fgsm_attack(model, criterion, data, target, epsilon):
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Generates adversarial examples by perturbing inputs in the direction
    of the gradient of the loss with respect to the input.
    
    Parameters
    ----------
    model : nn.Module
        The neural network model to attack
    criterion : nn.Module
        Loss function (e.g., nn.MSELoss())
    data : torch.Tensor
        Input data (batch_size, features)
    target : torch.Tensor
        True labels/targets
    epsilon : float
        Perturbation magnitude (controls attack strength)
    
    Returns
    -------
    perturbed_data : torch.Tensor
        Adversarial examples
    perturbation : torch.Tensor
        The perturbation added to the original data
    """
    # ensure data requires gradient
    data = data.clone().detach().requires_grad_(True)

    # forward pass
    output = model(data)

    # calculate loss
    loss = criterion(output.squeeze(-1), target)

    # zero all existing gradients
    model.zero_grad()

    # backward pass to get gradient w.r.t. input
    loss.backward()

    # collect the element wise sign of the data gradient
    data_grad = data.grad.data
    sign_data_grad = data_grad.sign()

    # create the perturbed data
    perturbation = epsilon * sign_data_grad
    perturbed_data = data + perturbation

    return perturbed_data.detach(), perturbation.detach()


def evaluate_robustness(model: nn.Module,
                        valid_loader: DataLoader,
                        criterion: nn.Module,
                        epsilon_values: list,
                        device: torch.device):
    """
    Evaluate model robustness against FGSM attacks at different epsilon values.
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    valid_loader : DataLoader
        Test/validation data
    criterion : nn.Module
        Loss function
    epsilon_values : list
        List of epsilon values to test
    device : torch.device
        Device to run on
        
    Returns
    -------
    results : dict
        Dictionary with epsilon values as keys and metrics as values
    """
    model.eval()
    model.to(device)

    results = {
        eps: {'loss': 0.0, 'predictions': [], 'targets': [], 'perturbed_data': []}
        for eps in epsilon_values
    }
    
    # clean data
    results[0.0] = {
        'loss': 0.0, 'predictions': [], 'targets': [], 'perturbed_data': []
    }

    with torch.no_grad():
        # evaluate on clean data first
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.squeeze(-1), target)

            results[0.0]['loss'] += loss.item()
            results[0.0]['predictions'].extend(output.cpu().numpy())
            results[0.0]['targets'].extend(target.cpu().numpy())

    results[0.0]['loss'] /= len(valid_loader)

    # evaluate on adversarial examples
    for epsilon in epsilon_values:
        if epsilon == 0.0:
            continue

        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)

            # generate adversarial examples
            perturbed_data, perturbation = fgsm_attack(model=model,
                                                       criterion=criterion,
                                                       data=data,
                                                       target=target,
                                                       epsilon=epsilon)
            
            # evaluate on perturbed data
            with torch.no_grad():
                output = model(perturbed_data)
                loss = criterion(output.squeeze(-1), target)

                results[epsilon]['loss'] += loss.item()
                results[epsilon]['predictions'].extend(output.cpu().numpy())
                results[epsilon]['targets'].extend(target.cpu().numpy())
                results[epsilon]['perturbed_data'].extend(perturbed_data.cpu().numpy())

        results[epsilon]['loss'] /= len(valid_loader)

    return results


def compare_model_robustness(fnn_model,
                             lnn_model,
                             valid_loader,
                             criterion,
                             epsilon_values,
                             scaler,
                             device):
    """
    Compare robustness of Feedforward NN vs Lipschitz NN.
    
    Parameters
    ----------
    fnn_model : nn.Module
        Feedforward neural network
    lnn_model : nn.Module
        Lipschitz neural network
    valid_loader : DataLoader
        Test/validation data
    criterion : nn.Module
        Loss function
    epsilon_values : list
        Attack strengths to test
    scaler : sklearn scaler
        Data scaler
    device : torch.device
        Device to run on
        
    Returns
    -------
    model1_results : dict
        Results for model1
    model2_results : dict
        Results for model2
    """
    print("Evaluating Feedforward Neural Network robustness...")
    fnn_results = evaluate_robustness(fnn_model,
                                      valid_loader,
                                      criterion,
                                      epsilon_values,
                                      device)
    
    print("Evaluating Lipschitz Neural Network robustness...")
    lnn_results = evaluate_robustness(lnn_model,
                                      valid_loader,
                                      criterion,
                                      epsilon_values,
                                      device)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('FGSM Attack: Feedforward NN vs Lipschitz NN', fontsize=16, fontweight='bold')
    
    # 1. Loss vs Epsilon
    ax1 = axes[0, 0]
    epsilons = [0.0] + epsilon_values
    fnn_losses = [fnn_results[eps]['loss'] for eps in epsilons]
    lnn_losses = [lnn_results[eps]['loss'] for eps in epsilons]
    
    ax1.plot(epsilons, fnn_losses, 'o-', label='Feedforward NN', linewidth=2, markersize=8)
    ax1.plot(epsilons, lnn_losses, 's-', label='Lipschitz NN', linewidth=2, markersize=8)
    ax1.set_xlabel('Epsilon (Attack Strength)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Loss Under Attack', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MAE vs Epsilon
    ax2 = axes[0, 1]
    fnn_maes = []
    lnn_maes = []
    
    for eps in epsilons:
        fnn_pred = np.array(fnn_results[eps]['predictions']).reshape(-1, 1)
        fnn_target = np.array(fnn_results[eps]['targets']).reshape(-1, 1)
        fnn_maes.append(mean_absolute_error(
            scaler.inverse_transform(fnn_target),
            scaler.inverse_transform(fnn_pred)
        ))
        
        lnn_pred = np.array(lnn_results[eps]['predictions']).reshape(-1, 1)
        lnn_target = np.array(lnn_results[eps]['targets']).reshape(-1, 1)
        lnn_maes.append(mean_absolute_error(
            scaler.inverse_transform(lnn_target),
            scaler.inverse_transform(lnn_pred)
        ))
    
    ax2.plot(epsilons, fnn_maes, 'o-', label='Feedforward NN', linewidth=2, markersize=8)
    ax2.plot(epsilons, lnn_maes, 's-', label='Lipschitz NN', linewidth=2, markersize=8)
    ax2.set_xlabel('Epsilon (Attack Strength)', fontsize=12)
    ax2.set_ylabel('MAE (Original Scale)', fontsize=12)
    ax2.set_title('Mean Absolute Error Under Attack', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Prediction degradation (relative to clean)
    ax3 = axes[1, 0]
    fnn_clean_mae = fnn_maes[0]
    lnn_clean_mae = lnn_maes[0]
    
    fnn_degradation = [(mae - fnn_clean_mae) / fnn_clean_mae * 100 for mae in fnn_maes]
    lnn_degradation = [(mae - lnn_clean_mae) / lnn_clean_mae * 100 for mae in lnn_maes]
    
    ax3.plot(epsilons, fnn_degradation, 'o-', label='Feedforward NN', linewidth=2, markersize=8)
    ax3.plot(epsilons, lnn_degradation, 's-', label='Lipschitz NN', linewidth=2, markersize=8)
    ax3.set_xlabel('Epsilon (Attack Strength)', fontsize=12)
    ax3.set_ylabel('Performance Degradation (%)', fontsize=12)
    ax3.set_title('Relative Performance Degradation', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 4. Sample predictions at highest epsilon
    ax4 = axes[1, 1]
    max_eps = max(epsilon_values)
    n_samples = min(50, len(fnn_results[max_eps]['predictions']))
    
    fnn_pred_scaled = scaler.inverse_transform(
        np.array(fnn_results[max_eps]['predictions'][:n_samples]).reshape(-1, 1)
    )
    lnn_pred_scaled = scaler.inverse_transform(
        np.array(lnn_results[max_eps]['predictions'][:n_samples]).reshape(-1, 1)
    )
    targets_scaled = scaler.inverse_transform(
        np.array(fnn_results[max_eps]['targets'][:n_samples]).reshape(-1, 1)
    )
    
    x_range = range(n_samples)
    ax4.plot(x_range, targets_scaled, 'k-', label='True Values', linewidth=2, alpha=0.7)
    ax4.plot(x_range, fnn_pred_scaled, 'o-', label='Feedforward NN', alpha=0.6, markersize=4)
    ax4.plot(x_range, lnn_pred_scaled, 's-', label='Lipschitz NN', alpha=0.6, markersize=4)
    ax4.set_xlabel('Sample Index', fontsize=12)
    ax4.set_ylabel('Volatility', fontsize=12)
    ax4.set_title(f'Predictions Under Strong Attack (Epsilon={max_eps})', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("ROBUSTNESS COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Epsilon':<10} {'FNN Loss':<15} {'LNN Loss':<15} {'FNN MAE':<15} {'LNN MAE':<15}")
    print("-"*80)
    
    for eps in epsilons:
        fnn_loss = fnn_results[eps]['loss']
        lnn_loss = lnn_results[eps]['loss']
        fnn_mae = fnn_maes[epsilons.index(eps)]
        lnn_mae = lnn_maes[epsilons.index(eps)]
        print(f"{eps:<10.3f} {fnn_loss:<15.6f} {lnn_loss:<15.6f} {fnn_mae:<15.6f} {lnn_mae:<15.6f}")
    
    print("\n" + "="*80)
    print("Performance degradation at maximum epsilon:")
    print(f"Feedforward NN: {fnn_degradation[-1]:.2f}%")
    print(f"Lipschitz NN: {lnn_degradation[-1]:.2f}%")
    print(f"Relative improvement: {(fnn_degradation[-1] - lnn_degradation[-1]):.2f}%")
    print("="*80)
    
    return fnn_results, lnn_results
