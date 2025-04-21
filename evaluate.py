import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

import trainer
import data_loader
import model

import time 

def test_fluid_simulation_model(model, test_loader, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    all_predictions, all_targets = [], []
    total_loss = 0.0
    total_time = 0.0  # Track total prediction time
    total_frames = 0  # Track total number of frames processed
    
    with torch.no_grad():
        prev_velocity = None  # Initialize prev_velocity as None

        for batch_idx, (batch_inputs, batch_targets) in enumerate(test_loader):
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            # Count frames in this batch (first dimension is batch size)
            batch_size = batch_inputs.size(0)
            total_frames += batch_size

            # Measure time before prediction
            start_time = time.time()
            # Forward pass (prediction)
            predictions = model(batch_inputs)
            # Measure time after prediction
            elapsed_time = time.time() - start_time
            total_time += elapsed_time  # Accumulate total time

            # Initialize prev_velocity for the first batch
            prev_velocity = torch.zeros_like(batch_targets) if batch_idx == 0 else prev_velocity[:batch_targets.size(0)]

            # Compute loss using prev_velocity
            batch_loss = trainer.fluid_simulation_loss(predictions, batch_targets, prev_velocity)
            total_loss += batch_loss.item()

            # Update prev_velocity for the next batch
            prev_velocity = predictions.detach()

            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())

    
    # Concatenate predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: (N, 2, H, W)
    all_targets = np.concatenate(all_targets, axis=0)  # Shape: (N, 2, H, W)
    
    # Compute evaluation metrics
    metrics_dict = compute_fluid_metrics(all_predictions, all_targets)
    metrics_dict['average_loss'] = total_loss / len(test_loader)
    
    # Compute average time per batch and per frame
    metrics_dict['total_prediction_time'] = total_time
    metrics_dict['average_time_per_batch'] = total_time / len(test_loader)
    metrics_dict['average_time_per_frame'] = total_time / total_frames
        
    return metrics_dict, all_predictions, all_targets

def compute_velocity_magnitude(velocity_field):
    vx, vy = velocity_field[0], velocity_field[1]
    return np.sqrt(vx**2 + vy**2)

def visualize_velocity_predictions(predictions, targets, n_samples=5):
    n_samples = min(n_samples, len(predictions))

    fig, axes = plt.subplots(2, 3, figsize=(12, 3 * n_samples))
    i = 100

    pred_magnitude = compute_velocity_magnitude(predictions[i])
    # X-Velocity Component
    axes[0, 0].imshow(predictions[i, 0], cmap='coolwarm', origin="lower", interpolation="bilinear")
    axes[0, 0].set_title(f'Predicted vx (Sample {i+1})')
    axes[0, 0].axis("off")
    
    # Y-Velocity Component
    axes[0, 1].imshow(predictions[i, 1], cmap='coolwarm', origin="lower", interpolation="bilinear")
    axes[0, 1].set_title(f'Predicted vy (Sample {i+1})')
    axes[0, 1].axis("off")
    
    # Velocity Magnitude
    axes[0, 2].imshow(pred_magnitude, cmap='viridis', origin="lower", interpolation="bilinear")
    axes[0, 2].set_title(f'Predicted Velocity Magnitude (Sample {i+1})')
    axes[0, 2].axis("off")
    
    target_magnitude = compute_velocity_magnitude(targets[i])
    # X-Velocity Component
    axes[1, 0].imshow(targets[i, 0], cmap='coolwarm', origin="lower", interpolation="bilinear")
    axes[1, 0].set_title(f'Target vx (Sample {i+1})')
    axes[1, 0].axis("off")

    # Y-Velocity Component
    axes[1, 1].imshow(targets[i, 1], cmap='coolwarm', origin="lower", interpolation="bilinear")
    axes[1, 1].set_title(f'Target vy (Sample {i+1})')
    axes[1, 1].axis("off")

    # Velocity Magnitude
    axes[1, 2].imshow(target_magnitude, cmap='viridis', origin="lower", interpolation="bilinear")
    axes[1, 2].set_title(f'Target Velocity Magnitude (Sample {i+1})')
    axes[1, 2].axis("off")

    # plt.tight_layout()
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

def run_model_testing(model, test_loader, visualize=True):
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
    loaded_model = model.UNet()
    loaded_model.load_state_dict(checkpoint['model_state_dict'])

    # Run testing and visualization
    metrics_dict, predictions, targets = run_model_testing(loaded_model, test_loader, visualize=False)

    return metrics_dict, predictions, targets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", help = "Model file path", type = str, required=True)
    parser.add_argument("-d", "--data", help = "Set dataset directory", type = str, required=True)

    args = parser.parse_args()

    _, test_loader = data_loader.load(args.data)

    evaluate_model(args.model, test_loader)