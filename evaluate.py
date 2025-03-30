import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

import trainer

def test_fluid_simulation_model(model, test_loader, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    all_predictions, all_targets = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            # Forward pass
            predictions = model(batch_inputs)

            # Compute loss (ensure `fluid_simulation_loss` is defined)
            batch_loss = trainer.fluid_simulation_loss(predictions, batch_targets)
            total_loss += batch_loss.item()

            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())
    
    # Concatenate predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: (N, 2, H, W)
    all_targets = np.concatenate(all_targets, axis=0)  # Shape: (N, 2, H, W)

    # Compute evaluation metrics
    metrics_dict = compute_fluid_metrics(all_predictions, all_targets)
    metrics_dict['average_loss'] = total_loss / len(test_loader)
    
    return metrics_dict, all_predictions, all_targets

def compute_velocity_magnitude(velocity_field):
    vx, vy = velocity_field[0], velocity_field[1]
    return np.sqrt(vx**2 + vy**2)

def visualize_velocity_predictions(predictions, targets, n_samples=5):
    n_samples = min(n_samples, len(predictions))

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))

    for i in range(n_samples):
        pred_magnitude = compute_velocity_magnitude(predictions[i])
        print(pred_magnitude.shape)
        # X-Velocity Component
        axes[i, 0].imshow(predictions[i, 0], cmap='coolwarm', origin="lower", interpolation="bilinear")
        axes[i, 0].set_title(f'Predicted vx (Sample {i+1})')
        axes[i, 0].axis("off")
        
        # Y-Velocity Component
        axes[i, 1].imshow(predictions[i, 1], cmap='coolwarm', origin="lower", interpolation="bilinear")
        axes[i, 1].set_title(f'Predicted vy (Sample {i+1})')
        axes[i, 1].axis("off")
        
        # Velocity Magnitude
        axes[i, 2].imshow(pred_magnitude, cmap='viridis', origin="lower", interpolation="bilinear")
        axes[i, 2].set_title(f'Predicted Velocity Magnitude (Sample {i+1})')
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Ground truth
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))

    for i in range(n_samples):
        target_magnitude = compute_velocity_magnitude(targets[i])

        # X-Velocity Component
        axes[i, 0].imshow(targets[i, 0], cmap='coolwarm', origin="lower", interpolation="bilinear")
        axes[i, 0].set_title(f'Target vx (Sample {i+1})')
        axes[i, 0].axis("off")

        # Y-Velocity Component
        axes[i, 1].imshow(targets[i, 1], cmap='coolwarm', origin="lower", interpolation="bilinear")
        axes[i, 1].set_title(f'Target vy (Sample {i+1})')
        axes[i, 1].axis("off")

        # Velocity Magnitude
        axes[i, 2].imshow(target_magnitude, cmap='viridis', origin="lower", interpolation="bilinear")
        axes[i, 2].set_title(f'Target Velocity Magnitude (Sample {i+1})')
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

def compute_fluid_metrics(predictions, targets):
    metrics_dict = {}

    pred_magnitudes = np.array([compute_velocity_magnitude(pred) for pred in predictions])
    target_magnitudes = np.array([compute_velocity_magnitude(target) for target in targets])

    # Velocity magnitude errors
    metrics_dict['velocity_mse'] = np.mean((pred_magnitudes - target_magnitudes) ** 2)
    metrics_dict['velocity_rmse'] = np.sqrt(metrics_dict['velocity_mse'])
    metrics_dict['velocity_mae'] = np.mean(np.abs(pred_magnitudes - target_magnitudes))

    # Component-wise errors
    for i, comp in enumerate(['x', 'y']):
        comp_pred, comp_target = predictions[:, i], targets[:, i]
        metrics_dict[f'{comp}_velocity_mse'] = np.mean((comp_pred - comp_target) ** 2)
        metrics_dict[f'{comp}_velocity_mae'] = np.mean(np.abs(comp_pred - comp_target))

    # Overall MSE, RMSE, and MAE
    metrics_dict['mse'] = np.mean((predictions - targets) ** 2)
    metrics_dict['rmse'] = np.sqrt(metrics_dict['mse'])
    metrics_dict['mae'] = np.mean(np.abs(predictions - targets))

    return metrics_dict

def run_model_testing(model, test_loader, visialize=True):
    # Run testing
    metrics_dict, predictions, targets = test_fluid_simulation_model(model, test_loader)

    # Print metrics
    print("\nFluid Simulation Model Testing Results:")
    for metric, value in metrics_dict.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.6f}")

    # Visualize predictions
    visualize_velocity_predictions(predictions, targets)

    return metrics_dict, predictions, targets

def evaluate_model(model_path, test_loader):
    # Load model
    checkpoint = torch.load(model_path)
    model = trainer.UNet()  # Ensure this matches your model architecture
    model.load_state_dict(checkpoint['model_state_dict'])

    # Run testing and visualization
    metrics_dict, predictions, targets = run_model_testing(model, test_loader, visialize=False)

    return metrics_dict, predictions, targets