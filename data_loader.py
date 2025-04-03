import torch
import numpy as np
from torch.utils.data import DataLoader
import os

class FluidDataset(torch.utils.data.Dataset):
    def __init__(self, density, velocity, inflow=None, obstacle=None, normalize=True):
        self.density = torch.tensor(density, dtype=torch.float32)
        self.velocity = torch.tensor(velocity, dtype=torch.float32)  # (C, H, W)
        self.inflow = torch.tensor(inflow, dtype=torch.float32) if inflow is not None else None
        self.obstacle = torch.tensor(obstacle, dtype=torch.float32) if obstacle is not None else None
        self.normalize = normalize
        self.time_steps = 2

        # Ensure all tensors have the same spatial size
        self.height, self.width = self.velocity.shape[-2:]  # Get H, W from velocity

        # Normalize data
        if normalize:
            self.density_min, self.density_max = self.density.min(), self.density.max()
            self.velocity_min, self.velocity_max = self.velocity.min(), self.velocity.max()
            self.density = (self.density - self.density.min()) / (self.density.max() - self.density.min() + 1e-8)
            self.velocity = (self.velocity - self.velocity.min()) / (self.velocity.max() - self.velocity.min() + 1e-8)

            if self.inflow is not None:
                self.inflow = (self.inflow - self.inflow.min()) / (self.inflow.max() - self.inflow.min() + 1e-8)
            if self.obstacle is not None:
                self.obstacle = (self.obstacle - self.obstacle.min()) / (self.obstacle.max() - self.obstacle.min() + 1e-8)

    def __len__(self):
        return len(self.density) -self.time_steps
    
    def normalize_tensor(self, tensor, min_val, max_val):
        min_val = tensor.min()
        max_val = tensor.max()
        # if max_val - min_val < 1e-6:  # Prevent division by zero
        #     return torch.zeros_like(tensor)
        return (tensor - min_val) / (max_val - min_val)
    
    def __getitem__(self, idx):
        # Extract last time step
        density_input = self.density[idx + self.time_steps - 1]  # (1, H, W)
        velocity_input = self.velocity[idx + self.time_steps - 1]  # (2, H, W)

        # Reshape density to match (1, H, W)
        density_input = density_input.unsqueeze(0)  # Shape: (1, 128, 96)
        # Reshape velocity to match (2, H, W)
        velocity_input = velocity_input.permute(2, 0, 1)  # Shape: (2, 128, 96)

        # Normalize if needed
        if self.normalize:
            density_input = self.normalize_tensor(density_input, self.density_min, self.density_max)
            velocity_input = self.normalize_tensor(velocity_input, self.velocity_min, self.velocity_max)

        # Stack input → (3, H, W) = [vx, vy, density]
        input_tensor = torch.cat([velocity_input, density_input], dim=0)  # (3, H, W)

        # Target: Next velocity field
        velocity_target = self.velocity[idx + self.time_steps]  # (2, H, W)
        velocity_target = velocity_target.permute(2, 0, 1)  # (H, W, C) → (C, H, W)

        return input_tensor, velocity_target

def load(data_dir, batch_size=16):
    train_densities = []
    train_velocities = []

    test_densities = []
    test_velocities = []

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # Iterate through all .npz files in the directory
    for file_name in os.listdir(train_dir):
        if file_name.endswith(".npz"):
            file_path = os.path.join(train_dir, file_name)
            data = np.load(file_path)

            # Load density and velocity arrays
            train_densities.append(data['density'])
            train_velocities.append(data['velocity'])

    for file_name in os.listdir(test_dir):
        if file_name.endswith(".npz"):
            file_path = os.path.join(test_dir, file_name)
            data = np.load(file_path)

            # Load density and velocity arrays
            test_densities.append(data['density'])
            test_velocities.append(data['velocity'])

    # Concatenate all data along the first dimension
    train_densities = np.concatenate(np.array(train_densities), axis=0)
    train_velocities = np.concatenate(np.array(train_velocities), axis=0)
    test_densities = np.concatenate(np.array(test_densities), axis=0)
    test_velocities = np.concatenate(np.array(test_velocities), axis=0)

    # Concatenate all data along the first dimension
    train_densities = np.concatenate(train_densities, axis=0)  # (Total_N, 1, 128, 96)
    train_velocities = np.concatenate(train_velocities, axis=0)  # (Total_N, 3, 128, 96)
    test_densities = np.concatenate(test_densities, axis=0)  # (Total_N, 1, 128, 96)
    test_velocities = np.concatenate(test_velocities, axis=0)  # (Total_N, 3, 128, 96)

    # Keep only the first 2 velocity channels (vx, vy)
    train_velocities = train_velocities[:, :, :, :2]  # (Total_N, 2, 128, 96)
    test_velocities = test_velocities[:, :, :, :2]  # (Total_N, 2, 128, 96)

    # Create dataset
    train_dataset = FluidDataset(train_densities, train_velocities)
    test_dataset = FluidDataset(test_densities, test_velocities)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader